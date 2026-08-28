import sys
import types
import torch
import transformers.modeling_utils
import transformers.pytorch_utils
import transformers.tokenization_utils_base

def apply_compat_patches():
    # 1. Mock wandb to prevent broken wandb.proto protobuf errors on ImageReward.ReFL import
    if "wandb" not in sys.modules:
        try:
            import wandb
        except Exception:
            mock_wandb = types.ModuleType("wandb")
            mock_wandb.init = lambda *args, **kwargs: None
            mock_wandb.log = lambda *args, **kwargs: None
            mock_wandb.finish = lambda *args, **kwargs: None
            mock_wandb.watch = lambda *args, **kwargs: None
            mock_wandb.save = lambda *args, **kwargs: None
            sys.modules["wandb"] = mock_wandb
            
    # 2. Transformers backwards-compatibility for BLIP/ImageReward
    def apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *args):
        assert len(args) == 1
        arg = args[0]
        num_chunks = max(1, arg.shape[chunk_dim] // chunk_size)
        return torch.cat([forward_fn(x) for x in arg.chunk(num_chunks, dim=chunk_dim)], dim=chunk_dim)

    def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        mask = torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        return heads, index

    def prune_linear_layer(layer, index, dim=0):
        index = index.to(layer.weight.device)
        W = layer.weight.index_select(dim, index).clone().detach()
        if layer.bias is not None:
            if dim == 1:
                b = layer.bias.clone().detach()
            else:
                b = layer.bias[index].clone().detach()
        new_size = list(layer.weight.size())
        new_size[dim] = len(index)
        new_layer = torch.nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
        new_layer.weight.requires_grad = False
        new_layer.weight.copy_(W.contiguous())
        new_layer.weight.requires_grad = True
        if layer.bias is not None:
            new_layer.bias.requires_grad = False
            new_layer.bias.copy_(b.contiguous())
            new_layer.bias.requires_grad = True
        return new_layer

    for mod in (transformers.modeling_utils, transformers.pytorch_utils):
        if not hasattr(mod, "apply_chunking_to_forward"):
            setattr(mod, "apply_chunking_to_forward", apply_chunking_to_forward)
        if not hasattr(mod, "find_pruneable_heads_and_indices"):
            setattr(mod, "find_pruneable_heads_and_indices", find_pruneable_heads_and_indices)
        if not hasattr(mod, "prune_linear_layer"):
            setattr(mod, "prune_linear_layer", prune_linear_layer)

    # 3. Tokenizer backwards-compatibility for BertTokenizer in BLIP/ImageReward
    _orig_add_special_tokens = transformers.tokenization_utils_base.PreTrainedTokenizerBase.add_special_tokens

    def _patched_add_special_tokens(self, *args, **kwargs):
        res = _orig_add_special_tokens(self, *args, **kwargs)
        try:
            enc_id = self.convert_tokens_to_ids("[ENC]")
            if enc_id is not None and enc_id != getattr(self, "unk_token_id", -1):
                self.__dict__["additional_special_tokens_ids"] = [enc_id]
            else:
                self.__dict__["additional_special_tokens_ids"] = [30522]
        except Exception:
            self.__dict__["additional_special_tokens_ids"] = [30522]
        return res

    transformers.tokenization_utils_base.PreTrainedTokenizerBase.add_special_tokens = _patched_add_special_tokens

    _orig_getattr = getattr(transformers.tokenization_utils_base.PreTrainedTokenizerBase, "__getattr__", None)
    def _patched_getattr(self, key):
        if key == "additional_special_tokens_ids":
            ids = []
            if hasattr(self, "additional_special_tokens") and self.additional_special_tokens:
                try:
                    res = self.convert_tokens_to_ids(self.additional_special_tokens)
                    if isinstance(res, list) and res:
                        ids = res
                    elif isinstance(res, int) and res != getattr(self, "unk_token_id", -1):
                        ids = [res]
                except Exception:
                    pass
            if not ids:
                try:
                    enc_id = self.convert_tokens_to_ids("[ENC]")
                    if enc_id is not None and enc_id != getattr(self, "unk_token_id", -1):
                        ids = [enc_id]
                except Exception:
                    pass
            if not ids:
                ids = [30522]
            return ids
        if _orig_getattr is not None:
            return _orig_getattr(self, key)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    transformers.tokenization_utils_base.PreTrainedTokenizerBase.__getattr__ = _patched_getattr

    # 4. Direct patch to ImageReward BLIP init_tokenizer function
    try:
        import ImageReward.models.BLIP.blip as blip_mod
        import ImageReward.models.BLIP.blip_pretrain as blip_pretrain_mod
        from transformers import BertTokenizer

        def safe_init_tokenizer():
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            tokenizer.add_special_tokens({'bos_token':'[DEC]'})
            tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})
            enc_id = tokenizer.convert_tokens_to_ids('[ENC]')
            if enc_id is None or enc_id == tokenizer.unk_token_id:
                enc_id = 30522
            tokenizer.enc_token_id = enc_id
            tokenizer.additional_special_tokens_ids = [enc_id]
            return tokenizer

        blip_mod.init_tokenizer = safe_init_tokenizer
        blip_pretrain_mod.init_tokenizer = safe_init_tokenizer
    except Exception:
        pass

apply_compat_patches()
