# 📑 KẾ HOẠCH THỰC NGHIỆM TỔNG THỂ & ĐỐI CHỨNG KHOA HỌC
## PROJECT: Lookahead Sample Reward Guidance with Dimension-Free Smoothed Surrogate for Test-Time Scaling of Diffusion Models

---

## 📌 I. TỔNG QUAN ĐỀ TÀI & ĐỘNG LỰC NGHIÊN CỨU

### 1. Bối cảnh & Bài báo đối ứng
* **Bài báo mục tiêu:** *Lookahead Sample Reward Guidance for Test-Time Scaling of Diffusion Models* (ICML 2026 Submission / [arXiv:2602.03211](https://arxiv.org/abs/2602.03211)).
* **Mô hình nền tảng:** Stable Diffusion v1.5 (`runwayml/stable-diffusion-v1-5`) & Bộ benchmark chuẩn **GenEval (553 prompts)**.
* **Môi trường thực thi:** Kaggle 2x Nvidia Tesla T4 GPU (16GB VRAM $\times 2$).

### 2. Vấn đề khoa học cốt lõi (Core Research Problem)
Phương pháp **LiDAR** sử dụng bộ giải nhanh (DPM-Solver 5 bước) để sinh trước $n=50$ hạt ước lượng nhanh $\hat{x}_0$, sau đó dùng hàm Reward nơ-ron $r(\hat{x}_0)$ (như ImageReward) để tính trọng số Softmax dẫn đường cho quá trình sinh ảnh chính thức (DDIM 50 bước).

Chúng tôi phát hiện ra **3 lỗ hổng toán học chí mạng** của LiDAR gốc:
1. **Độ nhạy cực đoan với sai số bộ giải do hệ số Lipschitz lớn (High-Lipschitz Sensitivity):** Các mạng nơ-ron chấm điểm có gradient cục bộ rất lớn ($L_r \gg 1$). Sai số hình học của bộ giải $e = \|\hat{x}_0 - x_0\|_2$ bị khuếch đại làm sai lệch nghiêm trọng giá trị Reward $|r(\hat{x}_0) - r(x_0)| \le L_r \|e\|_2$, đảo lộn thứ hạng các hạt và dẫn đường sai hướng.
2. **Hiện tượng "Winner-Takes-All" làm sụp đổ Entropy Softmax:** Độ dốc dựng đứng của Reward làm phân bố trọng số $w_i \propto \exp(\lambda r(\hat{x}_0^{(i)}))$ bị co cụm vào 1-2 hạt cá biệt, triệt tiêu tính đa dạng của tập hạt ($H(w) \to 0$).
3. **Sự rung lắc của trường Vector dẫn đường (Guidance Field Instability):** Vector dẫn đường $\mathbf{g}_t$ bị giật cục và đổi hướng liên tục giữa các bước thời gian $t$ và $t-\Delta t$ (Cosine Similarity thấp).

### 3. Đề xuất giải pháp: Smoothed Surrogate Reward Guidance
Thay thế hàm Reward thô $r(\hat{x}_0)$ bằng hàm Reward làm mượt Gauss:
$$\bar{r}_\sigma(z) = \mathbb{E}_{\xi \sim \mathcal{N}(0, I)} [r(z + \sigma \xi)]$$
Có **chặn Lipschitz không phụ thuộc số chiều (Dimension-Free Lipschitz Bound)**:
$$\|\nabla \bar{r}_\sigma(z)\|_2 \le \frac{M}{\sigma} \sqrt{\frac{2}{\pi}}$$
giúp triệt tiêu hoàn toàn sự khuếch đại sai số, ổn định phân bố Softmax và làm trơn trường vector dẫn đường.

---

## 🔬 II. TOÀN BỘ CÁC THÍ NGHIỆM: ĐÃ CHẠY & SẼ CHẠY

```text
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   🎯 TOÀN BỘ LỘ TRÌNH THỰC NGHIỆM                                 │
└─────────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                  │
         ┌────────────────────────────────────────┴────────────────────────────────────────┐
         ▼                                                                                 ▼
┌─────────────────────────────────────────────────┐               ┌─────────────────────────────────────────────────┐
│  📌 PHẦN A: TÁI LẬP BENCHMARK BÀI BÁO GỐC       │               │  🔬 PHẦN B: CHỨNG MINH ĐỘT PHÁ LÝ THUYẾT (OURS) │
│     (LiDAR DPM-5 / n=50 trên 553 GenEval)       │               │     (Smoothed Surrogate & Chặn Lipschitz)       │
└────────────────────────┬────────────────────────┘               └────────────────────────┬────────────────────────┘
                         │                                                                 │
         ┌───────────────┴───────────────┐                         ┌───────────────────────┼───────────────────────┐
         ▼                               ▼                         ▼                       ▼                       ▼
┌─────────────────┐             ┌─────────────────┐       ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ 【Pha 1】       │             │ 【Pha 2】       │       │ 【Bài Test 1】  │     │ 【Bài Test 2】  │     │ 【Bài Test 3】  │
│ Sinh 50 hạt     │  ───────►   │ Sinh ảnh Đích   │       │ Đo Sai số Solver│     │ Đo Sụp đổ       │     │ Đo Độ Trơn      │
│ Lookahead 5 bước│             │ DDIM 50 bước    │       │ & Lipschitz     │     │ Entropy H(w)    │     │ Trường Vector   │
│                 │             │                 │       │                 │     │                 │     │                 │
│ [ĐÃ XONG 100%]  │             │ [ĐANG CHẠY ~2h] │       │ [~20 Phút]      │     │ [~1 Phút]       │     │ [~2 Phút]       │
└─────────────────┘             └────────┬────────┘       └─────────────────┘     └─────────────────┘     └─────────────────┘
                                         │
                                         ▼
                                ┌─────────────────┐
                                │ 【Step 4 & 5】  │
                                │ Chấm điểm Bảng 2│
                                │ & Đóng gói ZIP  │
                                └─────────────────┘
```


---

## 📊 III. CHI TIẾT TỪNG PHA THÍ NGHIỆM

### 1. PHẦN A: TÁI LẬP BENCHMARK CHUẨN CỦA BÀI BÁO GỐC

#### 🔹 Thí nghiệm 1: Sinh Hạt Ước Lượng Nhanh (Pha 1 - Lookahead Sampling)
* **Mục tiêu:** Sinh không gian hạt lookahead thô cho toàn bộ 553 GenEval Prompts.
* **Cấu hình:**
  * Bộ giải: DPM-Solver 5 bước (`--num_inference_steps=5`).
  * Số hạt: $n = 50$ hạt / prompt (`--num_particles=50`).
  * Seed: $100$. Khoảng thời gian: $t \in [1000, 200]$.
  * Phân bổ phần cứng: 2 GPU song song (GPU 0: Prompt chẵn; GPU 1: Prompt lẻ).
* **Kết quả đầu ra:** 553 thư mục `00000` $\rightarrow$ `00552` chứa file `latent.pt` ($50 \times 4 \times 64 \times 64$) và điểm ImageReward ban đầu.
* **Trạng thái:** ✅ **ĐÃ HOÀN THÀNH $100\%$** (Được tái sử dụng vĩnh viễn ở các bước sau qua Input).

---

#### 🔹 Thí nghiệm 2: Sinh Ảnh Đích Dẫn Đường LiDAR (Pha 2 - Target Sampling)
* **Mục tiêu:** Tái lập chính xác bảng số liệu Table 2 & Table 8 trong bài báo gốc.
* **Cấu hình:**
  * Bộ giải: DDIM 50 bước (`--num_inference_steps=50`, `--eta=1.0`).
  * Số hạt đích: $N = 4$ hạt / prompt (`--num_particles=4`).
  * Siêu tham số dẫn đường: Hệ số tỷ lệ $\lambda = 5000$, Scale $s = 12.5$, Top-$k = 50$, $t_{\text{end}} = 200$.
  * Phân bổ phần cứng: 2 GPU song song (`--num_splits=2`, `--overwrite=True`).
  * Thời gian chạy: **~1.5 - 2.0 tiếng**.
* **Chỉ số đo lường & Đối chứng chuẩn mực:**
  1. **ImageReward (Trung bình 4 hạt):** Mục tiêu đạt **$0.378 \sim 0.384$** (so với baseline unguided là $-0.125$).
  2. **ImageReward (Hạt tốt nhất Best-of-4 Rank 1):** Mục tiêu đạt **$0.969$** (Table 8).
  3. **CLIP Score:** Mục tiêu đạt **$0.278$** (so với baseline $0.269$).
  4. **HPS v2.1 Score:** Mục tiêu đạt **$0.272 \sim 0.275$** (so với baseline $0.270$).
* **Đầu ra:** 2,212 bức ảnh thành phẩm, file `verification_report.json` và file nén `LiDAR_Full_Experiment.zip`.
* **Trạng thái:** ⏳ **ĐANG THỰC THI (Sắp hoàn tất)**.

---

### 2. PHẦN B: BỘ 3 BÀI TEST CHỨNG MINH ĐỘT PHÁ LÝ THUYẾT (THE 3 GOLDEN TESTS)

#### 🧪 Thí nghiệm 3 (Test 1): Đo Sai Số Bộ Giải & Hệ Số Lipschitz Thực Nghiệm (Theorem 1 Proof)
* **Mục tiêu:** Chứng minh toán học rằng hàm Reward gốc có hệ số Lipschitz rất lớn làm đảo lộn thứ hạng, trong khi Smoothed Surrogate bảo toàn thứ hạng hạt chuẩn xác.
* **Phương pháp đo:**
  * Với mỗi prompt, sinh 10 hạt từ **DPM-Solver 5 bước ($\hat{x}_0$)** và **DDIM 50 bước chuẩn ($x_0$)** từ cùng 1 seed.
  * Đo khoảng cách sai số hình học: $e_i = \|\hat{x}_0^{(i)} - x_0^{(i)}\|_2$.
  * Đo tỷ số Lipschitz thực nghiệm:
    $$\hat{L}_{\text{LiDAR}} = \frac{|r(\hat{x}_0) - r(x_0)|}{\|e\|_2} \quad \text{vs} \quad \hat{L}_{\text{Ours}} = \frac{|\bar{r}_\sigma(\hat{x}_0) - \bar{r}_\sigma(x_0)|}{\|e\|_2}$$
  * Đo hệ số tương quan thứ hạng **Kendall's $\tau$**:
    * $\tau_{\text{LiDAR}} = \text{KendallTau}(r(\hat{x}_0), r(x_0))$
    * $\tau_{\text{Ours}} = \text{KendallTau}(\bar{r}_\sigma(\hat{x}_0), \bar{r}_\sigma(x_0))$
* **Cấu hình tối ưu:** 100 Prompts đại diện (phân tầng đều trên 6 nhóm) $\times$ 10 hạt $\times$ 2 GPU $\rightarrow$ **Chạy trong 20 phút**.
* **Kỳ vọng kết quả:**
  * $\tau_{\text{LiDAR}} \approx 0.15 \sim 0.30$ (Rất thấp do nhiễu bộ giải).
  * $\tau_{\text{Ours}} \ge \mathbf{0.80 \sim 0.92}$ (Rất cao, thứ hạng hạt tốt nhất được bảo toàn).
  * Chặn Lipschitz: $\hat{L}_{\text{Ours}} \le \frac{M}{\sigma} \sqrt{\frac{2}{\pi}}$ được thỏa mãn $100\%$.
* **Đầu ra:** Biểu đồ `test_1_solver_error_robustness.png`.

---

#### 🧪 Thí nghiệm 4 (Test 2): Đo Hiện Tượng Sụp Đổ Entropy Softmax (Winner-Takes-All Analysis)
* **Mục tiêu:** Chứng minh phân bố Softmax của LiDAR gốc bị co cụm cực đoan vào 1 hạt bị thổi phồng điểm ảo, trong khi Smoothed Surrogate duy trì được sự đa dạng của tập hạt.
* **Phương pháp đo:**
  * Tính phân bố xác suất trọng số $w_i = \text{Softmax}(\lambda r(\hat{x}_0^{(i)}))$ và $w_i^{\text{ours}} = \text{Softmax}(\lambda \bar{r}_\sigma(\hat{x}_0^{(i)}))$.
  * Đo **Shannon Entropy**:
    $$H(w) = - \sum_{i=1}^n w_i \log w_i$$
  * Đo **Perplexity (Số lượng hạt hữu hiệu)**: $\text{PPL}(w) = \exp(H(w))$.
* **Cấu hình:** Quét trên dải nhiệt độ $\lambda \in [100, 10000]$. Thời gian chạy: **~1 phút**.
* **Kỳ vọng kết quả:**
  * LiDAR gốc: $H(w)$ sụp đổ về gần $0$ khi $\lambda \ge 2000$ (chỉ có 1 hạt chiếm $99.9\%$ trọng số).
  * Smoothed Surrogate: $H(w)$ duy trì ở mức cao ($> 2.5$), giữ được $\ge 15-20$ hạt đóng góp vào quá trình dẫn đường.
* **Đầu ra:** Biểu đồ `test_2_entropy_collapse.png`.

---

#### 🧪 Thí nghiệm 5 (Test 3): Đo Độ Trơn Và Sự Rung Lắc Của Trường Vector Dẫn Đường
* **Mục tiêu:** Chứng minh vector dẫn đường của LiDAR bị giật cục giữa các bước thời gian, trong khi Smoothed Surrogate tạo ra trường vector liên tục và trơn tru.
* **Phương pháp đo:**
  * Tính vector dẫn đường $\mathbf{g}_t$ tại từng bước thời gian $t \in [1000, 200]$.
  * Đo độ tương đồng hướng **Cosine Similarity** giữa 2 bước liên tiếp:
    $$\text{CosineSim}(\mathbf{g}_t, \mathbf{g}_{t-\Delta t}) = \frac{\langle \mathbf{g}_t, \mathbf{g}_{t-\Delta t} \rangle}{\|\mathbf{g}_t\|_2 \|\mathbf{g}_{t-\Delta t}\|_2}$$
* **Thời gian chạy:** **~2 phút**.
* **Kỳ vọng kết quả:**
  * LiDAR gốc: Cosine Similarity dao động mạnh ($0.40 \sim 0.70$), vector đổi hướng đột ngột.
  * Smoothed Surrogate: Cosine Similarity tiệm cận trơn tru $\mathbf{0.92 \sim 0.98}$.
* **Đầu ra:** Biểu đồ `test_3_guidance_field_smoothness.png`.

---

#### 🧪 Thí nghiệm 6 (Pha 3): Sinh Ảnh Thực Tế Với Smoothed Surrogate LiDAR
* **Mục tiêu:** So sánh chất lượng ảnh sinh ra giữa LiDAR gốc vs Smoothed Surrogate LiDAR trên các mức làm mượt $\sigma \in \{0.02, 0.05, 0.10\}$.
* **Chỉ số đo lường:** ImageReward, CLIP, HPS v2.1, và Tỷ lệ thắng đánh giá thị giác con người (User Preference Win Rate).
* **Kỳ vọng:** Smoothed Surrogate LiDAR triệt tiêu các vết nhiễu (noise artifacts), tăng độ sắc nét và vượt điểm ImageReward của LiDAR gốc ($> 0.40$).

---

## 📈 IV. BẢNG TỔNG HỢP TIẾN ĐỘ & PHÂN BỔ THỜI GIAN

| STT | Tên Thí Nghiệm | Mục Tiêu Khoa Học | Phần Cứng | Thời Gian | Tình Trạng |
|:---:|---|---|:---:|:---:|:---:|
| **1** | **Pha 1: Lookahead (DPM-5/n=50)** | Tạo không gian hạt thô cho 553 prompts | 2x GPU T4 | ~1.2 tiếng | ✅ **ĐÃ XONG 100%** |
| **2** | **Pha 2: Target LiDAR (DDIM-50)** | Tái lập Table 2 ($0.378$) & Table 8 ($0.969$) | 2x GPU T4 | ~1.5 - 2.0 tiếng | ⏳ **ĐANG CHẠY** |
| **3** | **Step 4: Chấm điểm Benchmark** | Đánh giá ImageReward, CLIP, HPS v2.1 | 1 GPU | ~3 phút | ⏳ **KẾ TIẾP** |
| **4** | **Step 5: Đóng gói ZIP 1-Click** | Nén dữ liệu và xuất link tải | CPU | ~1 phút | ⏳ **KẾ TIẾP** |
| **5** | **Test 1: Lipschitz & Sai số Solver** | Chứng minh Theorem 1 & Kendall $\tau$ | 2x GPU T4 | ~20 phút | 📅 **SẮP CHẠY** |
| **6** | **Test 2: Sụp đổ Softmax Entropy** | Chứng minh chống suy biến đa dạng hạt | GPU/CPU | ~1 phút | 📅 **SẮP CHẠY** |
| **7** | **Test 3: Trơn Trường Vector** | Chứng minh ổn định Cosine Similarity | GPU/CPU | ~2 phút | 📅 **SẮP CHẠY** |
| **8** | **Pha 3: Sinh ảnh Smoothed LiDAR** | Đo chất lượng ảnh sinh ra với $\bar{r}_\sigma$ | 2x GPU T4 | ~1.5 tiếng | 📅 **SẮP CHẠY** |

---

## 📑 V. SẢN PHẨM & TÀI LIỆU PHỤC VỤ BÁO CÁO (DELIVERABLES)

1. **Bảng Báo Cáo Đối Chứng Chính Thức:** File `verification_report.json` và bảng tổng kết so sánh trực tiếp với Table 2/8 của bài báo gốc.
2. **Bộ 3 Biểu Đồ Minh Họa Định Lý:**
   * `figures/test_1_solver_error_robustness.png` (Biểu đồ phân bố sai số & Kendall's $\tau$).
   * `figures/test_2_entropy_collapse.png` (Đường cong suy biến Shannon Entropy theo $\lambda$).
   * `figures/test_3_guidance_field_smoothness.png` (Biểu đồ độ mượt Cosine Similarity qua các bước khuếch tán).
3. **Kho Lưu Trữ Mẫu Dữ Liệu:** File `LiDAR_Full_Experiment.zip` chứa trọn vẹn 2,212 bức ảnh thành phẩm độ phân giải cao và toàn bộ logs.
4. **Mã Nguồn Mở Hoàn Chỉnh:** Repository GitHub `https://github.com/leekwanreal/Noisy-Reward` có đầy đủ tài liệu hướng dẫn và mã nguồn tự động hóa.
