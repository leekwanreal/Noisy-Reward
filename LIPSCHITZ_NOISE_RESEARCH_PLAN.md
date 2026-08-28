# 📋 Kế hoạch Nghiên cứu: Phân tích & Chứng minh Điểm yếu của Phương pháp Thêm Nhiễu (Randomized Smoothing) để Chặn Hệ số Lipschitz trong Reward Guidance

Tài liệu này vạch ra lộ trình nghiên cứu toán học và thực nghiệm toàn diện nhằm phân tích bản chất, chứng minh các giới hạn lý thuyết và đo đạc thực nghiệm các điểm yếu khi áp dụng kỹ thuật **Thêm Nhiễu (Gaussian Perturbation / Randomized Smoothing)** vào Lookahead Samples nhằm làm mịn và chặn hệ số Lipschitz của hàm Reward.

---

## 1. 🎯 Giả thuyết & Cơ sở Toán học của Ý tưởng

### Cơ chế hoạt động:
Khi thêm nhiễu Gauss $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$ vào mẫu Lookahead $x_0$:
$$\tilde{x}_0 = x_0 + \epsilon$$
Hàm thưởng trung bình (Smoothed Reward) trở thành tích chập Gauss:
$$\bar{r}_\sigma(x) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, \sigma^2 I)} [r(x + \epsilon)]$$

Theo **Bổ đề Stein (Gaussian Stein's Lemma)**:
$$\nabla_x \bar{r}_\sigma(x) = \frac{1}{\sigma^2} \mathbb{E}_{\epsilon} [\epsilon \cdot r(x + \epsilon)]$$
Nếu điểm Reward bị chặn trong khoảng $[-R_{\max}, R_{\max}]$, thì chuẩn Gradient có chặn trên giải tích:
$$\|\nabla_x \bar{r}_\sigma(x)\|_2 \le \frac{R_{\max} \sqrt{d}}{\sigma} = L_{\sigma} < \infty$$
👉 **Ưu điểm rõ ràng:** Hệ số Lipschitz $L_\sigma$ **chắc chắn bị chặn trên** và tỷ lệ nghịch với độ lệch chuẩn nhiễu $\sigma$.

---

## 2. ⚠️ 4 Điểm yếu Cốt lõi Cần Chứng minh (Theoretical & Empirical Weaknesses)

### 🔴 Điểm yếu 1: Đánh đổi Độ lệch - Phương sai (Reward Degradation & Ranking Inconsistency)
- **Bản chất:** Các hàm Reward thị giác (ImageReward, HPS v2.1) cực kỳ nhạy cảm với các chi tiết tần số cao (High-frequency details như ánh sáng, vân nổi, ngón tay, văn bản trong GenEval).
- **Hậu quả:** Khi thêm nhiễu $\sigma$, ảnh bị mờ/hạt $\rightarrow$ Điểm thưởng $r(x_0 + \epsilon)$ tụt giảm nghiêm trọng và làm **đảo lộn thứ tự xếp hạng (Ranking Inconsistency)** của 50 hạt Lookahead. Hạt thực sự sắc nét có thể bị chấm điểm thấp hơn hạt trơn nhẵn!

### 🔴 Điểm yếu 2: Suy giảm Tỷ số Tín hiệu trên Nhiễu (SNR Collapse & Information Loss)
- Khi tăng $\sigma$ để ghìm Lipschitz xuống mức an toàn, tín hiệu gradient của Reward bị san phẳng:
  $$\lim_{\sigma \to \infty} \bar{r}_\sigma(x) = \text{Const} \implies \nabla \bar{r}_\sigma(x) \to \mathbf{0}$$
- Khi đó, lực dẫn đường của LiDAR bị vô hiệu hóa hoàn toàn, mô hình mất khả năng tối ưu hóa thẩm mỹ.

### 🔴 Điểm yếu 3: Bùng nổ Chi phí Tính toán (Monte Carlo Variance Dilemma)
- Nếu chỉ lấy **1 mẫu nhiễu đơn lẻ** ($\tilde{x} = x + \epsilon$): Điểm reward $\hat{r}$ có phương sai ngẫu nhiên cực lớn, gây rung lắc quỹ đạo khử nhiễu.
- Nếu muốn ước lượng kỳ vọng $\bar{r}_\sigma(x)$ chính xác: Phải lấy $M$ mẫu nhiễu ($M \ge 10$) $\rightarrow$ **Thời gian chấm điểm ImageReward tăng gấp $M$ lần** (Ví dụ: Pha 1 từ 25s vọt lên $250\text{s}$ / prompt)!

### 🔴 Điểm yếu 4: Mâu thuẫn Bước thời gian ở Pha 2 (Late-step Guidance Mismatch)
- Ở các bước khử nhiễu cuối ($t \to 0$), ảnh $x_t$ đang rất sạch. Việc dùng vector mồi bị nhiễu $\tilde{x}_0$ sẽ vô tình **kéo ảnh về trạng thái có nhiễu**, làm mất độ sắc nét và gây hiện tượng bệt màu (Blurry Artifacts).

---

## 3. 🧪 Kế hoạch Thực nghiệm 3 Giai đoạn để Làm rõ Điểm yếu

```
[Giai đoạn 1: Đo đạc Lipschitz vs Độ lệch Reward]
                      │
                      ▼
[Giai đoạn 2: Đo độ tương quan Xếp hạng (Kendall's Tau)]
                      │
                      ▼
[Giai đoạn 3: Chạy Testbed Sinh ảnh Thực tế trên GenEval]
```

### 🔬 Giai đoạn 1: Đo Đánh đổi giữa Hệ số Lipschitz ($L_\sigma$) và Điểm Reward ($r$)
- **Mục tiêu:** Xây dựng đường cong Pareto thể hiện: Khi $\sigma$ tăng để giảm Lipschitz $L$, điểm Reward $r$ tụt dốc như thế nào.
- **Thực hiện:**
  - Chọn dải $\sigma \in [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]$.
  - Với mỗi $\sigma$, đo $L_\sigma = \|\nabla \bar{r}_\sigma(z)\|_2$ và điểm thưởng trung bình $\bar{r}$.
  - Vẽ đồ thị 2 trục $Y$: Trục trái là $L_\sigma$ (giảm), trục phải là ImageReward (suy giảm).

### 🔬 Giai đoạn 2: Đo Độ Xáo trộn Thứ tự Xếp hạng (Kendall's $\tau$ & Spearman's $\rho$)
- **Mục tiêu:** Chứng minh toán học rằng việc thêm nhiễu làm sai lệch thứ tự Top-K hạt tốt nhất.
- **Thực hiện:**
  - Lấy 50 hạt Lookahead của cùng 1 prompt.
  - Xếp hạng gốc: $\text{Rank}_{\text{clean}} = \text{argsort}(r(x_0^{(1)}), \dots, r(x_0^{(50)}))$.
  - Xếp hạng sau khi thêm nhiễu: $\text{Rank}_{\text{noisy}} = \text{argsort}(r(x_0^{(1)} + \epsilon), \dots, r(x_0^{(50)} + \epsilon))$.
  - Tính hệ số tương quan thứ bậc:
    $$\tau(\sigma) = \text{Kendall\_Tau}(\text{Rank}_{\text{clean}}, \text{Rank}_{\text{noisy}})$$
  - Chứng minh khi $\sigma > 0.05$, $\tau$ giảm mạnh về $0$, chứng tỏ hạt được chọn làm mồi dẫn đường thực chất là hạt ngẫu nhiên, không còn đại diện cho chất lượng cao!

### 🔬 Giai đoạn 3: Đánh giá Chất lượng Sinh ảnh Thực tế trên GenEval (Table 2)
- **Mục tiêu:** Chạy thực nghiệm trọn vẹn cả 2 pha trên tập prompt GenEval với các mức $\sigma$ khác nhau để xem chỉ số định lượng cuối cùng:
  - $\sigma = 0.0$ (LiDAR gốc)
  - $\sigma = 0.05$ (Nhiễu nhẹ)
  - $\sigma = 0.1$ (Nhiễu vừa)
  - $\sigma = 0.2$ (Nhiễu mạnh)
- So sánh các chỉ số: $IR$ (ImageReward), $CLIP$, $HPS$ và thời gian chạy.

---

## 4. 📂 Các Công cụ & File Mã nguồn Sẽ Triển khai

1. `experiments/analyze_lipschitz_noise_tradeoff.py`:
   - Script tự động quét dải $\sigma$ và tính $L_\sigma$, Kendall's Tau $\tau$, và Mean Reward.
2. `experiments/plot_tradeoff_curves.py`:
   - Tự động xuất các biểu đồ học thuật chuẩn phong cách ICML/NeurIPS (Matplotlib + Seaborn vector PDF/PNG).
3. `experiments/modified_noisy_lookahead.py`:
   - Module tích hợp cờ `--noise_sigma` vào Lookahead Sampling để chạy thử nghiệm Pha 1 & Pha 2 có thêm nhiễu.

---

## 5. 💡 Luận điểm Kết luận cho Bài báo / Luận văn của Bạn

Bằng cách chỉ ra rõ ràng điểm yếu của phương pháp thêm nhiễu (Randomized Smoothing):
> *"Mặc dù việc thêm nhiễu Gauss vào không gian ẩn giúp chặn trên hệ số Lipschitz về mặt lý thuyết, nhưng nó phá vỡ thứ tự tương quan phần thưởng (Ranking Inconsistency) và làm sụp đổ tỷ số tín hiệu trên nhiễu (SNR Collapse). Điều này đặt ra yêu cầu cấp thiết cho một phương pháp mới điều hướng gradient bị chặn mà không làm suy biến phân phối hạt thăm dò ban đầu."*

$\rightarrow$ Luận điểm này sẽ làm cho **phương pháp đóng góp chính của bạn trở nên vô cùng thuyết phục và vượt trội!**
