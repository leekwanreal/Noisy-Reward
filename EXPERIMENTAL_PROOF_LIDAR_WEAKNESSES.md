# Quy trình Thực nghiệm Chuẩn: Bộ 3 Bài Test Chứng minh Tính Cần thiết của Phương pháp Smoothed Surrogate (RS)

Tài liệu này xác lập quy trình thực nghiệm gồm **3 bài test chuẩn xác 100% theo khung lý thuyết Dimension-Free Lipschitz Bound**, khớp trực tiếp với script thực thi độc lập [`test_lidar_vs_smoothed_surrogate.py`](file:///c:/Users/Admin/Desktop/Deep%20Learning%20Research/Noisy%20Reward/Diffusion-LiDAR-Sampling/test_lidar_vs_smoothed_surrogate.py).

---

## 1. BÀI TEST 1: Đo Khả năng Kháng Sai số Bộ giải (Solver Error Robustness & Theorem 1)

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

### B. Hàm Thực thi trong Code:
- Hàm `run_test_1_solver_robustness(pipe, vae, ir_model, prompt_list, sigma, num_particles)`.

### C. Quy trình và Công thức tính toán chi tiết:
1. **Sinh cặp hạt đối chứng:** 
   - 5 bước DPM-Solver ($\hat{\mathbf{x}}_0^i$)
   - 50 bước DDIM chuẩn ($\mathbf{x}_0^i$) từ cùng một `seed` ngẫu nhiên.
2. **Tính sai số hình học:**
   $$\|\mathbf{e}_i\|_2 = \|\hat{\mathbf{x}}_0^i - \mathbf{x}_0^i\|_2$$
3. **Tính điểm thưởng:**
   - LiDAR gốc: $r_{\text{LiDAR}} = r(\hat{\mathbf{x}}_0^i)$
   - Phương pháp bạn: $\bar{r}_\sigma(\hat{\mathbf{x}}_0^i) = \frac{1}{M} \sum_{m=1}^M r(\hat{\mathbf{x}}_0^i + \boldsymbol{\xi}_m)$ với $M=4$.
4. **Đo sai số điểm và tương quan thứ bậc:**
   $$\Delta r = |r(\hat{\mathbf{x}}_0^i) - r(\mathbf{x}_0^i)|$$
   $$\tau = \text{Kendall\_Tau}(\text{Rank}_{\text{5-step}}, \text{Rank}_{\text{50-step}})$$

---

## 2. BÀI TEST 2: Đo Khả năng Kháng Sụp đổ Trọng số Softmax (Softmax Mode Collapse Prevention)

### A. Cơ sở Lý thuyết (Theo Slide 2 của Bạn):
Trọng số dẫn đường được tính theo công thức:

$$
w_i^r \propto \exp\left( \lambda \bar{r}_\sigma(\hat{\mathbf{x}}_0^i) - \frac{\|\mathbf{x}_t - \hat{\mathbf{x}}_0^i\|^2}{2\sigma_t^2} \right)
$$

- **LiDAR gốc:** $r(\hat{\mathbf{x}}_0^i)$ có các đỉnh gai nhọn làm hàm $\exp(\lambda r)$ bị bão hòa One-Hot $\implies$ $95\%$ trọng số dồn vào đúng 1 hạt (Best-of-1 Trap).
- **Phương pháp của Bạn:** $\bar{r}_\sigma$ làm phẳng các đỉnh gai nhọn $\implies$ Hàm Softmax phân bổ mượt mà trên toàn bộ $n=50$ hạt.

### B. Hàm Thực thi trong Code:
- Hàm `run_test_2_softmax_entropy(num_particles=50, num_steps=50)`.

### C. Quy trình và Công thức tính toán chi tiết:
1. Tại mỗi bước khử nhiễu $t \in [1000, 0]$:
   - Tính vector trọng số Softmax $w_i^r$ cho cả 2 phương pháp.
2. **Tính Entropy Shannon của vector trọng số:**
   $$H(w^r) = - \sum_{i=1}^{50} w_i^r \log_2(w_i^r + 10^{-12})$$
3. **Kỳ vọng số liệu:**
   - LiDAR gốc: $H(w^r) < 0.2\text{ bits}$ (Sụp đổ One-Hot).
   - Phương pháp của bạn: $H(w^r) \approx 4.0 \sim 5.0\text{ bits}$ (Phân bổ mượt mà).

---

## 3. BÀI TEST 3: Đo Độ Ổn định Lipschitz của Trường Vector Dẫn đường (Guidance Field Stability)

### A. Cơ sở Lý thuyết (Theo Slide 2 của Bạn):
Vector dẫn đường giải tích:

$$
\mathbf{g}_t = \sum_{i=1}^n (w_i^r - w_i) \frac{\hat{\mathbf{x}}_0^i}{\sigma_t^2}
$$

- Khi thêm nhiễu vi mô $\delta$ ($\|\delta\|_2 = 10^{-3}$) vào trạng thái $\mathbf{x}_t$, độ nhạy của vector $\mathbf{g}_t$ được chặn trên bởi hệ số Lipschitz $L_\sigma$:

$$
\left\| \frac{\partial \mathbf{g}_t}{\partial \mathbf{x}_t} \right\| \le C \cdot L_\sigma = C \frac{\Delta r}{\sigma \sqrt{2\pi}} < \infty
$$

### B. Hàm Thực thi trong Code:
- Hàm `run_test_3_guidance_stability(num_particles=50, delta_eps=0.001)`.

### C. Quy trình và Công thức tính toán chi tiết:
1. Tại các mốc bước khử nhiễu $t = 800, 600, 400, 200$:
   - Tính vector gốc: $\mathbf{g}_t = \mathbf{g}_t(\mathbf{x}_t)$.
   - Tính vector khi bị nhiễu: $\mathbf{g}_t' = \mathbf{g}_t(\mathbf{x}_t + \delta)$ với $\|\delta\|_2 = 0.001$.
2. **Đo Độ ổn định góc quay (Cosine Stability Ratio):**
   $$\text{CosSim}(\mathbf{g}_t, \mathbf{g}_t') = \frac{\langle \mathbf{g}_t, \mathbf{g}_t' \rangle}{\|\mathbf{g}_t\|_2 \|\mathbf{g}_t'\|_2}$$
3. **Kỳ vọng số liệu:**
   - LiDAR gốc: $\text{CosSim} < 0.5$ (Vector bị bẻ ngoặt hướng).
   - Phương pháp của bạn: $\text{CosSim} \ge \mathbf{0.98 \sim 0.99}$ (Kháng nhiễu tuyệt đối).

---

## 4. 🚀 Hướng dẫn Chạy Thực nghiệm trên Google Colab / GPU

Để chạy toàn bộ Bộ 3 Bài Test và tự động xuất ảnh biểu đồ + file JSON kết quả, bạn chỉ cần chạy lệnh sau trên Colab:

```bash
!python test_lidar_vs_smoothed_surrogate.py --num_prompts 10 --num_particles 20 --sigma 0.05
```

### 📁 Kết quả Xuất ra:
- **Biểu đồ 3 bài test:** `experiments/test_results/golden_3_tests_comparison.png`
- **File số liệu tổng hợp JSON:** `experiments/test_results/summary_results.json`
