
import os
import json
import math
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
from transformers import BertTokenizer
import random
import clip


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path, 0o777)

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer


import glob


class gen_lookahead_samples(Dataset):
    def __init__(self, directory, top_k):
        print("top_k:", top_k)
        
        # Build candidate directories to scan
        candidate_dirs = []
        if os.path.exists(directory):
            candidate_dirs.append(directory)
        for cand in [
            directory,
            os.path.join("Lookahead_samples", directory),
            f"/kaggle/working/LiDAR_Experiment/Lookahead_samples/{directory}",
            f"/kaggle/working/Lookahead_samples/{directory}",
            f"/content/drive/MyDrive/LiDAR_Experiment/Lookahead_samples/{directory}",
            f"/content/LiDAR_Experiment/Lookahead_samples/{directory}",
        ]:
            if os.path.exists(cand) and cand not in candidate_dirs:
                candidate_dirs.append(cand)
                
        # Also find any input directories matching Lookahead_samples
        for cand in glob.glob(f"/kaggle/input/**/Lookahead_samples/{directory}", recursive=True):
            if os.path.exists(cand) and cand not in candidate_dirs:
                candidate_dirs.append(cand)
        for cand in glob.glob(f"/kaggle/input/**/{directory}", recursive=True):
            if os.path.exists(cand) and os.path.isdir(cand) and cand not in candidate_dirs:
                candidate_dirs.append(cand)

        data = {}
        for d_path in candidate_dirs:
            if not os.path.exists(d_path) or not os.path.isdir(d_path):
                continue
            prompt_dirs = sorted([d for d in os.listdir(d_path) if os.path.isdir(os.path.join(d_path, d)) and d.isdigit()])
            for d in prompt_dirs:
                p_idx = int(d)
                if p_idx in data:
                    continue  # already loaded
                latent_path = os.path.join(d_path, d, "samples", "latent.pt")
                results_path = os.path.join(d_path, d, "results.json")
                if not os.path.exists(latent_path) or not os.path.exists(results_path):
                    continue
                try:
                    latent = torch.load(latent_path, map_location="cpu")
                    with open(results_path, "r", encoding="utf-8") as f:
                        label = json.load(f)
                    reward = label["ImageReward"]["result"]
                    prompt = label["prompt"]

                    k = min(top_k, len(latent))
                    latents = [latent[i] for i in range(k)]
                    rewards = [reward[i] for i in range(k)]
                    prompts = [prompt[i] for i in range(k)]
                    datapoint = {"latents": torch.stack(latents), "rewards": torch.tensor(rewards), "prompts": prompts}
                    data[p_idx] = datapoint
                except Exception as e:
                    print(f"Warning loading lookahead prompt {d} from {d_path}: {e}")
                    
        print(f"📦 Đã nạp thành công Lookahead samples cho {len(data)} prompts!")
        self.data = data


