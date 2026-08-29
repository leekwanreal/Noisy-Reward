# 🧠 TOÀN BỘ BỐI CẢNH DỰ ÁN & LỊCH SỬ PHÁT TRIỂN (README_CONTEXT.md)

---

## 📌 1. Tổng Quan Dự Án & Mục Tiêu Nghiên Cứu

* **Bài báo nghiên cứu đối ứng:** *Lookahead Sample Reward Guidance for Test-Time Scaling of Diffusion Models* (ICML 2026 Submission / [arXiv:2602.03211](https://arxiv.org/abs/2602.03211)).
  - Toàn văn 27 trang của bài báo đã được trích xuất sạch tại file: [`LIDAR_PAPER_EXTRACTED.txt`](LIDAR_PAPER_EXTRACTED.txt).
* **Repository:** `https://github.com/leekwanreal/Noisy-Reward.git`
* **Mục tiêu nghiên cứu:**
  1. **Tái lập Thực nghiệm Benchmark (SD v1.5 LiDAR DPM-5 / n=50 trên 553 GenEval Prompts - Bảng 2 & Bảng 8):** Đo đạc các chỉ số chất lượng sinh ảnh: *ImageReward (IR - Mean & Best-of-4)*, *CLIP-Score*, *HPS v2.1*, và *GenEval*.
  2. **Phát hiện & Chứng minh Toán học Lỗ hổng Cốt lõi của LiDAR:** Chỉ ra các điểm yếu nghiêm trọng của phương pháp LiDAR gốc khi sử dụng mô hình chấm điểm nhạy cảm/nhiễu trên các mẫu thô ước lượng từ bộ giải nhanh.
  3. **Đề xuất Giải pháp Cải tiến (Smoothed Surrogate Reward Guidance):** Xây dựng hàm Reward mượt $\bar{r}_\sigma(\hat{x}_0)$ có **chặn Lipschitz không phụ thuộc số chiều (Dimension-Free Lipschitz Bound)**.
  4. **Xây dựng Bộ 3 Bài Test Lý thuyết Độc lập (The 3 Golden Tests):** Đo lường và vẽ biểu đồ so sánh định lượng giữa LiDAR gốc và phương pháp Smoothed Surrogate.
  5. **Tối ưu hóa Thực thi Đa GPU trên Kaggle & Colab:** Xây dựng cơ chế chạy song song 2x GPU T4 (giảm 50% thời gian), tự động khôi phục dữ liệu phiên cũ và chạy ngầm tắt máy ("Save & Run All").

---

## 🔬 2. Lý Thuyết & Lỗ Hổng Cốt Lõi Của LiDAR Gốc

LiDAR sử dụng một bộ giải nhanh (như DPM-Solver 5 bước) để sinh trước $n=50$ hạt ước lượng nhanh $\hat{x}_0^{(i)}$, sau đó dùng mô hình Reward $r(\hat{x}_0^{(i)})$ để tính trọng số Softmax dẫn đường cho quá trình sinh ảnh chính thức (50 bước). Ba điểm yếu lý thuyết chí mạng được phát hiện gồm:

1. **Vấn đề 1 — Độ nhạy cực cao với Sai số Bộ giải (Solver Discretization Error Sensitivity):**
   * Do bước giải nhanh $\Delta t$ lớn, hạt ước lượng $\hat{x}_0$ có sai số hình học so với hạt thật $x_0$: $e = \hat{x}_0 - x_0$.
   * Do hàm Reward mạng nơ-ron không mượt, độ lệch điểm số bị khuếch đại:
     $$|r(\hat{x}_0) - r(x_0)| \le L_r \|\hat{x}_0 - x_0\|_2$$
     với $L_r$ cục bộ rất lớn, làm đảo lộn thứ hạng và dẫn đường sai hướng.
2. **Vấn đề 2 — Hiện tượng "Winner-Takes-All" trong Softmax Resampling (Entropy Collapse):**
   * Phân bố xác suất $w_i \propto \exp(\lambda r(\hat{x}_0^{(i)}))$ bị co cụm cực đoan vào 1-2 hạt bị overestimation ảo.
   * Làm sụt giảm Shannon Entropy $H(w)$, triệt tiêu tính đa dạng của tập hạt.
3. **Vấn đề 3 — Sự Rung lắc của Trường Vector Dẫn đường (Guidance Field Instability):**
   * Vector dẫn đường $\nabla_x \log \sum w_i \mathcal{N}(x; \mu_i, \sigma_i^2 I)$ bị giật cục giữa các bước thời gian $t$ và $t+\Delta t$, làm giảm Cosine Similarity và giảm độ chân thực của ảnh thành phẩm.

---

## 💡 3. Đề Xuất Giải Pháp: Smoothed Surrogate Reward Guidance

Chúng tôi đề xuất thay thế $r(\hat{x}_0)$ bằng hàm Reward làm mượt Gauss (Gaussian Smoothed Surrogate):
$$\bar{r}_\sigma(z) = \mathbb{E}_{\xi \sim \mathcal{N}(0, I)} [r(z + \sigma \xi)]$$

### 📐 Định Lý Chặn Lipschitz Không Phụ Thuộc Số Chiều (Dimension-Free Lipschitz Theorem):
Nếu mô hình Reward bị chặn biên độ $|r(x)| \le M$, thì hàm $\bar{r}_\sigma(z)$ khả vi vô hạn và gradient của nó bị chặn nghiêm ngặt:
$$\|\nabla \bar{r}_\sigma(z)\|_2 \le \frac{M}{\sigma} \sqrt{\frac{2}{\pi}}$$
* **Ý nghĩa đột phá:** Chặn trên này **hoàn toàn độc lập với số chiều không gian Latent $d = 4 \times 64 \times 64 = 16.384$**, triệt tiêu hoàn toàn hiện tượng khuếch đại sai số do số chiều lớn, làm ổn định xác suất Softmax và trường dẫn đường.

---

## 🧪 4. Bộ 3 Bài Test Thực Nghiệm Độc Lập (3 Golden Tests)

Toàn bộ 3 bài test được tách riêng thành script độc lập [`test_lidar_vs_smoothed_surrogate.py`](test_lidar_vs_smoothed_surrogate.py):

* **TEST 1 — Kháng Sai số Bộ giải & Kiểm chứng Chặn Lipschitz:**
  - Đo sai số hình học $\|\hat{x}_0 - x_0\|_2$ giữa DPM 5 bước và DDIM 50 bước.
  - So sánh độ lệch điểm $|\Delta r_{\text{LiDAR}}|$ vs $|\Delta r_{\text{Ours}}|$.
  - Đo hệ số tương quan thứ hạng Kendall's $\tau$ (phương pháp đề xuất giữ thứ hạng ổn định hơn vượt trội).
* **TEST 2 — Ổn định Phân bố Softmax & Bảo tồn Đa dạng Mẫu:**
  - Đo Shannon Entropy $H(w)$ và Perplexity $P(w) = \exp(H(w))$ trên phân bố trọng số của 50 hạt Lookahead.
  - Chứng minh phương pháp Smoothed Surrogate ngăn chặn hiện tượng Entropy Collapse.
* **TEST 3 — Kháng Rung lắc Vector Dẫn đường (Guidance Field Lipschitz Stability):**
  - Đo Cosine Similarity giữa vector dẫn đường $\mathbf{g}_t$ và $\mathbf{g}_{t-\Delta t}$.
  - Chứng minh trường gradient của phương pháp đề xuất mượt mà và liên tục hơn.
* **Kết quả đầu ra:** Tự động xuất biểu đồ trực quan 3 khung hình **`golden_3_tests_comparison.png`** và file dữ liệu **`summary_results.json`** phục vụ chèn trực tiếp vào bài báo.

---

## ⚡ 5. Tối Ưu Hóa Hạ Tầng: Kaggle Dual-GPU & Colab

1. **Chạy Song Song 2x GPU T4 trên Kaggle (Tăng tốc 2x):**
   - Bổ sung tham số `--num_splits` (mặc định 1) và `--split_idx` (mặc định 0) vào `lookahead_sampling.py` và `LiDAR_sampling.py`.
   - GPU 0 xử lý các prompt chỉ số chẵn ($0, 2, 4, \dots$); GPU 1 xử lý các prompt chỉ số lẻ ($1, 3, 5, \dots$).
   - Rút ngắn toàn bộ thời gian chạy cả 2 Pha trên 553 prompts từ **~7.6 tiếng xuống còn ~3.8 tiếng**.
   - Tương thích tự động với mọi số lượng GPU (1 GPU, 2 GPU, N GPU).
2. **Cơ Chế Tự Động Khôi Phục & Nối Tiếp (Session Fallback & Auto-Resume):**
   - Step 1 của Notebook tự động quét và giải nén mọi file ZIP hoặc thư mục từ `/kaggle/input/` sang `/kaggle/working/`.
   - Hàm `is_lookahead_complete()` và `is_target_complete()` kiểm tra đa tầng (Working + Input + Drive + Local) để **bỏ qua các prompt đã làm trong $0.001\text{s}$**.
3. **Cờ Bắt Buộc Sinh Mới (`--overwrite`):**
   - Cho phép người dùng chạy lại Pha 2 sinh ảnh đích mới tinh trong khi vẫn tận dụng được 100% dữ liệu Pha 1 cũ nằm trong `/kaggle/input/` (Read-Only).
4. **Cơ Chế Chạy Ngầm Tắt Máy (Kaggle Save & Run All):**
   - Cho phép người dùng bấm "Save Version" $\rightarrow$ "Save & Run All (Commit)" rồi tắt máy tính. Hệ thống tự hoàn thành, tính điểm Bảng 2, chạy 3 bài test và nén kết quả thành `LiDAR_Full_Experiment.zip`.

---

## 🛠️ 6. Lịch Sử Gỡ Lỗi & Tinh Chỉnh Toàn Diện (Debugging History)

| STT | Hiện tượng lỗi | Nguyên nhân gốc rễ | Giải pháp đã thực hiện |
|---|---|---|---|
| **1** | `ImportError: cannot import name 'EncoderDecoderCache' from 'transformers'` | Thư viện `peft` trong môi trường Kaggle Python 3.12 yêu cầu class mà `transformers==4.38.2` chưa có. | Tạo file bản vá gốc `compat_patch.py` tự động monkeypatch mock class `EncoderDecoderCache`, `apply_chunking_to_forward` trước khi import diffusers/peft. |
| **2** | `ModuleNotFoundError: No module named 'image_reward_utils'` | Import tương đối bị lỗi khi gọi từ thư mục cha trong script `rewards.py`. | Thêm `try...except` import đa đường dẫn và chèn `sys.path` thư mục `fkd_diffusers`. |
| **3** | Báo "Tổng số ảnh sinh ra: 0 ảnh" ở Step 4 | Code chỉ tìm ảnh cấp 1 `00000/*.png` (bị lọc file `grid.png`), trong khi ảnh hạt nằm ở thư mục con `00000/samples/*.png`. | Sửa hàm glob tìm kiếm chính xác bên trong `os.path.join(p_dir, "samples", "*.png")`. |
| **4** | Pha 1 và Pha 2 chạy vèo qua trong 0.1s mà không sinh ảnh | Định nghĩa hàm `is_lookahead_complete()` bị đặt thụt lề bên trong hàm `main()`, khiến vòng lặp `for prompt_idx...` bị hiểu là thân hàm con và không bao giờ được `main()` gọi. | Đưa định nghĩa 2 hàm kiểm tra ra ngoài mức module (toàn cục) và căn chỉnh thụt lề chuẩn xác vòng lặp trong `main()`. |
| **5** | `RuntimeError: Input type (c10::Half) and bias type (float) should be the same` tại `vae.decode()` | Pipeline SD nạp ở `float16` nhưng VAE lại được nạp độc lập ở `float32`. | Đồng bộ sử dụng trực tiếp `vae = pipe.vae` và viết hàm `decode_latents(latents, vae, pipe)` tự động ép kiểu và chia batch (chunk) an toàn. |
| **6** | Pha 2 bị sinh lại từ đầu dù dữ liệu cũ đã có | Mã nguồn gốc LiDAR gắn thêm đuôi giờ ngẫu nhiên `_{cur_time}` vào tên thư mục, làm tạo thư mục mới mỗi lần chạy. | Cố định tên thư mục `Target_samples/50_1.0_4_50_200_5000_12.5_100` và bổ sung quét tiền tố linh hoạt. |
| **7** | Bấm Run All trong phiên mới bị chạy lại | Dữ liệu phiên trước nằm trong `/kaggle/input/`, hàm kiểm tra cũ chỉ nhìn `/kaggle/working/`. | Thêm cơ chế 2 lớp: Đồng bộ toàn bộ dữ liệu từ Input sang Working ở Step 1 + Bổ sung Fallback quét trực tiếp từ `/kaggle/input/`. |
| **8** | `AttributeError: 'ImageReward' object has no attribute 'score_batched'` tại `test_lidar_vs_smoothed_surrogate.py` | Hàm `RM.load()` nạp class gốc `ImageReward` (chỉ có `.score()`), thiếu method xử lý batch `score_batched` có trong wrapper `IRSMC`. | Chuyển sang `rm_load` từ `image_reward_utils` và bổ sung monkeypatch fallback `_universal_score_batched` trực tiếp trong `compat_patch.py`. |
| **9** | **Điểm ImageReward thực nghiệm bị thấp (0.1845 vs 0.378 trong bài báo)** | `lookahead_datasets.py` cũ lưu dữ liệu vào `list` tuần tự `self.data = []`. Khi chạy 2 GPU, thứ tự thư mục chẵn/lẻ bị đảo lộn khiến Pha 2 bốc nhầm hạt lookahead của prompt khác để dẫn đường. | Chuyển `self.data` sang cấu trúc **Dictionary `self.data[p_idx]`** ánh xạ chính xác 1-1 theo `prompt_idx`, sửa `fkd_pipeline_sd.py` dùng tra cứu dictionary an toàn. |
| **10** | `NameError: name 'glob' is not defined` trong `lookahead_sampling.py` & `LiDAR_sampling.py` | Hàm `is_lookahead_complete()` và `is_target_complete()` gọi `glob.glob()` nhưng thiếu `import glob` ở đầu file, và chỉ kiểm tra duy nhất đường dẫn `/kaggle/input/`. | Bổ sung `import glob` và tổng quát hóa bộ quét đa tầng (Kaggle input, Colab drive, Relative path, Local). |
| **11** | Pha 2 bị tự động SKIP khi input phiên cũ chứa cả Target cũ (Read-Only) | `/kaggle/input/` là read-only không thể xóa, khiến `is_target_complete()` luôn thấy file cũ và bỏ qua Pha 2 dù muốn sinh lại. | Thêm cờ `--overwrite` vào `LiDAR_sampling.py`. Khi truyền cờ này, code bỏ qua kiểm tra hoàn thành và sinh mới 100% ảnh đích vào `/kaggle/working/Target_samples`. |
| **12** | `torch.OutOfMemoryError: CUDA out of memory` tại Test 1 VAE decode | `num_particles=20` truyền cả 20 hạt vào UNet cùng lúc, sau đó `decode_latents` giải mã VAE một lúc 10 ảnh ($512\times 512$), ngốn vượt quá 15GB VRAM của GPU T4. | Chia nhỏ sinh hạt UNet thành micro-batches `batch_size=4`, giải mã VAE `chunk_size=2` với `detach().cpu()`, bật `vae.enable_slicing()` và `vae.enable_tiling()`, dùng `torch.inference_mode()`. |

---

## 📊 7. Đối Chiếu Kết Quả Chuẩn Theo Bảng 2 & Bảng 8 Bài Báo

| Cấu hình Thực nghiệm | ImageReward (Mean 4 hạt) | ImageReward (Best-of-4 Rank 1) | CLIP Score | HPS v2.1 | GenEval |
|---|:---:|:---:|:---:|:---:|:---:|
| **Vanilla SD v1.5 (DDIM 50 bước)** | $-0.125$ | $-0.125$ | $0.269$ | $0.270$ | $0.423$ |
| **Universal Guidance (SD v1.5)** | $+0.047$ | $+0.047$ | $0.272$ | $0.268$ | $0.426$ |
| **DATE (SD v1.5)** | $+0.165$ | $+0.165$ | $0.274$ | $0.271$ | $0.449$ |
| **LiDAR DPM-5 / n=50 (Table 2)** | **$+0.378 \sim 0.384$** | — | **$0.278$** | **$0.272 \sim 0.275$** | **$0.475$** |
| **LiDAR Best-of-4 / n=50 (Table 8)** | — | **$+0.969$** | **$0.282$** | **$0.293$** | **$0.585$** |

---

## 🚀 8. Hướng Dẫn Vận Hành Dự Án

### A. Vận hành trên Kaggle (2x GPU T4):
1. Import Notebook: `File` $\rightarrow$ `Import Notebook` $\rightarrow$ `URL`:
   ```text
   https://github.com/leekwanreal/Noisy-Reward/blob/main/run_lidar_kaggle.ipynb
   ```
2. Cấu hình bên phải giao diện:
   * **Accelerator:** `GPU T4 x2`
   * **Internet:** `On`
   * **Input:** Bấm `+ Add Input` $\rightarrow$ `Your Work` $\rightarrow$ Chọn output phiên trước (để tự động nhận diện và bỏ qua Pha 1 trong 0.001s).
3. Bấm **`Save Version`** $\rightarrow$ Chọn **`Save & Run All (Commit)`** $\rightarrow$ Bấm **`Save`** $\rightarrow$ Tắt máy tính.

### B. Chạy riêng Bộ 3 Bài Test Lipschitz:
```bash
python test_lidar_vs_smoothed_surrogate.py \
    --num_prompts=-1 \
    --num_particles=20 \
    --sigma=0.05 \
    --output_dir="experiments/test_results"
```

---
*Tài liệu được tổng hợp tự động và duy trì đồng bộ trực tiếp với kho mã nguồn `leekwanreal/Noisy-Reward`.*
