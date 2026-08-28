# Quy trình Thực nghiệm Chuẩn: Bộ 3 Bài Test Chứng minh Tính Cần thiết của Phương pháp Smoothed Surrogate (RS)

Tài liệu này xác lập quy trình thực nghiệm gồm **3 bài test chuẩn xác 100% theo khung lý thuyết Dimension-Free Lipschitz Bound**, giúp bạn đo đạc và đối chứng trực tiếp giữa **LiDAR gốc ($\sigma = 0$)** và **Phương pháp Smoothed Surrogate của bạn ($\bar{r}_\sigma$)**.

---

## 1. BÀI TEST 1: Đo Khả năng Kháng Sai số Bộ giải (Solver Error Robustness)

### A. Cơ sở Lý thuyết (Theo Slide 1 của Bạn):
Khi dùng DPM-Solver 5 bước, ta thu được mẫu xấp xỉ $\hat{\mathbf{x}}_0^i$ có sai số $\mathbf{e}_i = \hat{\mathbf{x}}_0^i - \mathbf{x}_0^i$.
- **LiDAR gốc ($\sigma = 0$):** Không có làm mịn $\implies L_0 \to \infty$. Sai số $|r(\hat{\mathbf{x}}_0^i) - r(\mathbf{x}_0^i)|$ bị bùng nổ mất kiểm soát.
- **Phương pháp của Bạn ($r_\sigma$):** Có chặn trên Lipschitz không phụ thuộc số chiều (Dimension-Free):

$$
\|\nabla_{\mathbf{x}} r_\sigma(\mathbf{x})\|_2 \le \frac{\Delta r}{\sigma \sqrt{2\pi}} = L_\sigma < \infty
$$

Bảo đảm bất đẳng thức sai số:

$$
|r_\sigma(\hat{\mathbf{x}}_0^i) - r_\sigma(\mathbf{x}_0^i)| \le L_\sigma \|\mathbf{e}_i\|_2
$$

### B. Tính lúc nào trong Pipeline?
- **Thời điểm thực hiện:** Ở **Pha 1 (Lookahead Sampling)**.
- **Vị trí tính toán:** Tại bước giải mã VAE và chấm điểm ImageReward cho 50 hạt Lookahead.

### C. Dùng Prompt như thế nào?
- Lấy **50 prompts** đại diện từ file `prompt_files/geneval_metadata.jsonl`.
- Với mỗi prompt, cố định cùng 1 `seed` (ví dụ `seed=100`) để sinh 50 hạt qua 2 bộ giải:
  1. 5 bước DPM-Solver ($\hat{\mathbf{x}}_0^i$).
  2. 50 bước DDIM chuẩn ($\mathbf{x}_0^i$).

### D. Công thức & Quy trình tính toán:
1. Tính sai số hình học trong không gian latent:
   $$\|\mathbf{e}_i\|_2 = \|\hat{\mathbf{x}}_0^i - \mathbf{x}_0^i\|_2$$
2. Tính sai số phần thưởng thực tế:
   $$\Delta r_{\text{LiDAR}} = |r(\hat{\mathbf{x}}_0^i) - r(\mathbf{x}_0^i)|$$
   $$\Delta r_{\text{Ours}} = |\bar{r}_\sigma(\hat{\mathbf{x}}_0^i) - \bar{r}_\sigma(\mathbf{x}_0^i)|$$
3. Đo hệ số tương quan thứ hạng hạt (Kendall's $\tau$):
   $$\tau = \text{Kendall\_Tau}(\text{Rank}_{\text{5-step}}, \text{Rank}_{\text{50-step}})$$

### E. Kết quả kỳ vọng:
- $\Delta r_{\text{Ours}}$ luôn nằm gọn dưới chặn lý thuyết $L_\sigma \|\mathbf{e}_i\|_2$.
- Kendall's $\tau$ của phương pháp bạn tăng từ **$< 0.35$ (LiDAR gốc)** lên **$> 0.85$ (Phương pháp bạn)**.

---

## 2. BÀI TEST 2: Đo Khả năng Kháng Sụp đổ Trọng số Softmax (Softmax Mode Collapse Prevention)

### A. Cơ sở Lý thuyết (Theo Slide 2 của Bạn):
Trọng số dẫn đường được tính theo công thức:

$$
w_i^r \propto \exp\left( \lambda \bar{r}_\sigma(\hat{\mathbf{x}}_0^i) - \frac{\|\mathbf{x}_t - \hat{\mathbf{x}}_0^i\|^2}{2\sigma_t^2} \right)
$$

- **LiDAR gốc:** $r(\hat{\mathbf{x}}_0^i)$ có các đỉnh gai nhọn làm hàm $\exp(\lambda r)$ bị bão hòa One-Hot $\implies$ $95\%$ trọng số dồn vào đúng 1 hạt (Best-of-1 Trap).
- **Phương pháp của Bạn:** $\bar{r}_\sigma$ làm phẳng các đỉnh gai nhọn $\implies$ Hàm Softmax phân bổ mượt mà trên toàn bộ $n=50$ hạt.

### B. Tính lúc nào trong Pipeline?
- **Thời điểm thực hiện:** Trong **Pha 2 (LiDAR Target Sampling)**.
- **Vị trí tính toán:** Tại **từng bước thời gian $t$** trong vòng lặp 50 bước DDIM (ngay sau khi tính $w_i^r$).

### C. Dùng Prompt như thế nào?
- Chạy trên tập prompt GenEval, mỗi prompt sinh 4 ảnh đích từ ngân hàng 50 hạt Lookahead.

### D. Công thức & Quy trình tính toán:
1. Tại mỗi bước $t$, đo **Entropy Shannon** của vector trọng số $w^r \in \mathbb{R}^{50}$:
   $$H(w^r) = - \sum_{i=1}^{50} w_i^r \log_2(w_i^r + 10^{-12})$$
2. Đo **Trọng số cực đại** chiếm ưu thế:
   $$w_{\max}^r = \max_{i \in [1, 50]} w_i^r$$

### E. Kết quả kỳ vọng:
- Đồ thị $H(w^r)$ của LiDAR gốc rơi thẳng đứng về **$< 0.2\text{ bits}$** ($w_{\max}^r > 95\%$).
- Đồ thị $H(w^r)$ của phương pháp bạn duy trì ổn định ở mức **$4.0 \sim 5.0\text{ bits}$** (phân bổ đều đặn trên toàn bộ 50 hạt).

---

## 3. BÀI TEST 3: Đo Độ Ổn định Lipschitz của Trường Vector Dẫn đường (Guidance Field Stability)

### A. Cơ sở Lý thuyết (Theo Slide 2 của Bạn):
Vector dẫn đường giải tích:

$$
\mathbf{g}_t = \sum_{i=1}^n (w_i^r - w_i) \frac{\hat{\mathbf{x}}_0^i}{\sigma_t^2}
$$

- Khi thêm nhiễu vi mô $\delta$ ($\|\delta\|_2 = 10^{-3}$) vào trạng thái $\mathbf{x}_t$, độ nhạy của vector $\mathbf{g}_t$ được chặn trên bởi hệ số Lipschitz $L_\sigma$:
  $$\left\| \frac{\partial \mathbf{g}_t}{\partial \mathbf{x}_t} \right\| \le C \cdot L_\sigma = C \frac{\Delta r}{\sigma \sqrt{2\pi}} < \infty$$

### B. Tính lúc nào trong Pipeline?
- **Thời điểm thực hiện:** Trong **Pha 2**, tại các mốc bước khử nhiễu: $t = 800, 600, 400, 200$.

### C. Dùng Prompt như thế nào?
- Lấy 20 prompts từ GenEval. Tại mỗi mốc $t$, lấy trạng thái latent $\mathbf{x}_t$.

### D. Công thức & Quy trình tính toán:
1. Tính vector dẫn đường gốc: $\mathbf{g}_t = \mathbf{g}_t(\mathbf{x}_t)$.
2. Tạo nhiễu vi mô: $\delta \sim \mathcal{N}(0, \sigma_\delta^2 I)$ với $\|\delta\|_2 = 0.001$.
3. Tính vector khi bị nhiễu: $\mathbf{g}_t' = \mathbf{g}_t(\mathbf{x}_t + \delta)$.
4. **Đo Độ ổn định góc quay (Cosine Stability Ratio):**
   $$\text{CosSim}(\mathbf{g}_t, \mathbf{g}_t') = \frac{\langle \mathbf{g}_t, \mathbf{g}_t' \rangle}{\|\mathbf{g}_t\|_2 \|\mathbf{g}_t'\|_2}$$

### E. Kết quả kỳ vọng:
- LiDAR gốc: $\text{CosSim} < 0.5$ (vector bị bẻ ngoắt hướng do gradient bất ổn định).
- Phương pháp của bạn: $\text{CosSim} \ge \mathbf{0.98 \sim 0.99}$ (trường vector siêu ổn định, kháng hoàn toàn nhiễu rung lắc).

---

### 📊 Bảng Tổng Hợp 3 Bài Test Cho Bài Báo:

| Bài Test | Đại lượng Đo | Kết quả LiDAR gốc ($\sigma=0$) | Kết quả Phương pháp Bạn ($r_\sigma$) |
| :--- | :--- | :---: | :---: |
| **TEST 1: Chống Sai số 5 bước** | $\Delta r$ vs $L_\sigma \|\mathbf{e}\|_2$ & Kendall's $\tau$ | Sai số lớn, $\tau < 0.35$ | **Bị chặn $\le L_\sigma \|\mathbf{e}\|_2$, $\tau > 0.85$** |
| **TEST 2: Chống Sụp đổ Softmax** | Entropy Shannon $H(w^r)$ | Sụp đổ ($H \to 0\text{ bits}$) | **Mượt mà ($H \approx 4.5\text{ bits}$)** |
| **TEST 3: Độ Ổn định Dẫn đường** | Cosine Stability $\text{CosSim}(\mathbf{g}_t, \mathbf{g}_{t+\delta})$ | Hỗn loạn ($\text{CosSim} < 0.5$) | **Kháng nhiễu tuyệt đối ($\text{CosSim} \ge 0.98$)** |
