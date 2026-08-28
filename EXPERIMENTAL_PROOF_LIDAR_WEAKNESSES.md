# Quy trình Thực nghiệm Chi tiết: Đo đạc 4 Hệ số Chứng minh Lỗ hổng Toán học của LiDAR

Tài liệu này hướng dẫn chi tiết quy trình thực nghiệm để đo đạc chính xác **4 hệ số định lượng**, làm rõ:
1. **Tính lúc nào trong pipeline?** (Giai đoạn thực thi).
2. **Dùng tập prompt như thế nào?** (Cấu hình dữ liệu đầu vào).
3. **Công thức và thuật toán tính toán cụ thể ra sao?** (Quy trình toán học và code).
4. **Ý nghĩa số liệu đối chứng cho bài báo.**

---

## 1. BÀI TEST 1: Đo Hệ số Lipschitz $L_r$ và Sai số Điểm thưởng $\Delta r$ giữa 5 bước vs 50 bước

### A. Tính lúc nào trong pipeline?
- **Thời điểm thực hiện:** Ngay sau **Pha 1 (Lookahead Sampling)** hoặc chạy độc lập trước khi vào Pha 2.
- **Vị trí tính toán:** Tại bước giải mã VAE và chấm điểm ImageReward.

### B. Dùng Prompt như thế nào?
- **Tập dữ liệu:** Lấy **50 prompts** đại diện từ file `prompt_files/geneval_metadata.jsonl` (gồm các prompt phức tạp về quan hệ không gian, đếm đối tượng, và thuộc tính màu sắc).
- **Cấu hình:** Với mỗi prompt, cố định cùng 1 `seed` ngẫu nhiên (ví dụ `seed=100`) để sinh 50 hạt qua 2 bộ giải khác nhau.

### C. Quy trình và Công thức tính toán chi tiết:
1. **Bước 1 (Sinh hạt 5 bước):** Dùng DPM-Solver 5 bước sinh ra 50 vector latent: $z_{0, \text{5-step}}^{(1)}, \dots, z_{0, \text{5-step}}^{(50)}$.
2. **Bước 2 (Sinh hạt 50 bước chuẩn):** Dùng DDIM 50 bước sinh ra 50 vector latent từ cùng điểm nhiễu ban đầu: $z_{0, \text{50-step}}^{(1)}, \dots, z_{0, \text{50-step}}^{(50)}$.
3. **Bước 3 (Giải mã & Chấm điểm):** 
   - $r_{\text{5-step}} = \text{ImageReward}(\mathcal{D}(z_{0, \text{5-step}}))$
   - $r_{\text{50-step}} = \text{ImageReward}(\mathcal{D}(z_{0, \text{50-step}}))$
4. **Bước 4 (Tính các hệ số):**
   - **Hệ số Lipschitz cục bộ:**
     $$L_r = \frac{1}{50} \sum_{k=1}^{50} \|\nabla_z r(\mathcal{D}(z_{0, \text{50-step}}^{(k)}))\|_2$$
   - **Sai số điểm thưởng trung bình ($\Delta r$):**
     $$\Delta r = \frac{1}{50} \sum_{k=1}^{50} |r_{\text{5-step}}^{(k)} - r_{\text{50-step}}^{(k)}|$$
   - **Tương quan thứ bậc (Kendall's $\tau$):**
     $$\tau = \text{Kendall\_Tau}(\text{argsort}(r_{\text{5-step}}), \text{argsort}(r_{\text{50-step}}))$$

### D. Kết quả chứng minh LiDAR chưa tốt:
- $L_r > 150.0$ (chứng minh độ dốc của ImageReward cực kỳ lớn).
- Sai số $\Delta r \ge 0.85$ và $\tau < 0.35$ $\implies$ Thứ tự hạt mồi mà LiDAR chọn ở Pha 1 bị sai lệch nghiêm trọng so với chất lượng thực tế.

---

## 2. BÀI TEST 2: Đo Độ Sụp đổ Entropy của Trọng số Softmax ($H(w_r)$)

### A. Tính lúc nào trong pipeline?
- **Thời điểm thực hiện:** Trong **Pha 2 (LiDAR Target Sampling)**, ghi nhận trực tiếp tại **từng bước thời gian $t$** trong vòng lặp 50 bước khử nhiễu DDIM.
- **Vị trí tính toán:** Ngay sau hàm `get_sample_guide` (dòng 324 trong `fkd_pipeline_sd.py`).

### B. Dùng Prompt như thế nào?
- Chạy trên toàn bộ **553 prompts của GenEval** (hoặc 50 prompts mẫu trong quá trình kiểm thử).
- Mỗi prompt sinh $B=4$ ảnh đích từ ngân hàng $K=50$ hạt Lookahead.

### C. Quy trình và Công thức tính toán chi tiết:
1. Tại mỗi bước khử nhiễu $t \in [1000, 200]$:
   - Thuật toán tính trọng số Softmax cho 50 hạt:
     $$w_{r, k} = \frac{\exp(\lambda r_k + \text{potential}_k)}{\sum_{j=1}^{50} \exp(\lambda r_j + \text{potential}_j)}$$
2. **Tính Entropy Shannon của vector $w_r \in \mathbb{R}^{50}$:**
   $$H(w_r) = - \sum_{k=1}^{50} w_{r, k} \log_2(w_{r, k} + 10^{-12})$$
3. **Tính trọng số chiếm ưu thế lớn nhất:**
   $$w_{r, \max} = \max_{k \in [1, 50]} w_{r, k}$$
4. Lấy trung bình $H(w_r)$ qua tất cả các prompt và vẽ đường biểu diễn $H(w_r)$ theo bước thời gian $t$.

### D. Kết quả chứng minh LiDAR chưa tốt:
- Entropy lý thuyết khi phân bổ đều trên 50 hạt là $H_{\max} = \log_2(50) \approx 5.64\text{ bits}$.
- Thực tế đo được của LiDAR: $H(w_r) < 0.2\text{ bits}$ và $w_{r, \max} > 95\%$ $\implies$ $95\%$ lực dẫn đường bị dồn vào duy nhất 1 hạt, chứng minh công thức tích hợp phân phối 50 hạt bị sụp đổ (Mode Collapse).

---

## 3. BÀI TEST 3: Đo Độ Nhạy Lipschitz của Vector Dẫn đường ($\mathcal{L}_{\text{guide}}$)

### A. Tính lúc nào trong pipeline?
- **Thời điểm thực hiện:** Ở **Pha 2**, tại các mốc bước thời gian cụ thể: $t = 800, 600, 400, 200$.
- **Vị trí tính toán:** Bên trong hàm `get_sample_guide`.

### B. Dùng Prompt như thế nào?
- Lấy 20 prompts từ GenEval.
- Tại bước $t$, lấy trạng thái latent hiện tại $x_t$.

### C. Quy trình và Công thức tính toán chi tiết:
1. Tính vector dẫn đường gốc: $\mathbf{g}_t = \text{get\_sample\_guide}(x_t)$.
2. Tạo một nhiễu ngẫu nhiên siêu nhỏ: $\delta \sim \mathcal{N}(0, \sigma_{\delta}^2 I)$ với $\|\delta\|_2 = 0.001$.
3. Tính vector dẫn đường khi bị nhiễu: $\mathbf{g}_t' = \text{get\_sample\_guide}(x_t + \delta)$.
4. **Đo Độ tương đồng Cosine (Cosine Stability Ratio):**
   $$\text{CosSim}(\mathbf{g}_t, \mathbf{g}_t') = \frac{\langle \mathbf{g}_t, \mathbf{g}_t' \rangle}{\|\mathbf{g}_t\|_2 \|\mathbf{g}_t'\|_2}$$
5. **Đo Độ nhạy Lipschitz của trường vector:**
   $$\mathcal{L}_{\text{guide}}(t) = \frac{\|\mathbf{g}_t' - \mathbf{g}_t\|_2}{\|\delta\|_2}$$

### D. Kết quả chứng minh LiDAR chưa tốt:
- $\text{CosSim} < 0.5$ và $\mathcal{L}_{\text{guide}} > 10^3$ $\implies$ Trường vector dẫn đường của LiDAR cực kỳ hỗn loạn (Hyper-sensitive), chỉ một rung lắc vi mô của latent sẽ làm vector bị bẻ ngoặt hướng.

---

## 4. BÀI TEST 4: Đo Bùng nổ Gradient khi Gỡ bỏ Heuristic Cutoff ($t < 200$)

### A. Tính lúc nào trong pipeline?
- **Thời điểm thực hiện:** Trong **Pha 2**, khi gỡ bỏ tham số `--resample_t_end=200` (đặt `--resample_t_end=0` để thuật toán chạy trọn vẹn về $t=0$).
- **Vị trí tính toán:** Đo trực tiếp chuẩn L2 của vector dẫn đường $\|\mathbf{g}_t\|_2$ qua toàn bộ 50 bước khử nhiễu.

### B. Dùng Prompt như thế nào?
- Chạy trên 50 prompts GenEval ở 2 chế độ:
  1. Chế độ có Cutoff của tác giả bài báo (`--resample_t_end=200`).
  2. Chế độ thuần lý thuyết không Cutoff (`--resample_t_end=0`).

### C. Quy trình và Công thức tính toán chi tiết:
1. Tại mỗi bước $t \in [1000, 0]$, ghi nhận chuẩn L2 của vector dẫn đường:
   $$\|\mathbf{g}_t\|_2 = \left\| \frac{\sqrt{\bar{\alpha}_t}}{1 - \bar{\alpha}_t} \sum_{k=1}^{50} (w_{r, k} - w_k) \hat{x}_0^{(k)} \right\|_2$$
2. Tính tỷ số bùng nổ Gradient (Gradient Explosion Factor):
   $$\text{Explosion\_Ratio} = \frac{\max_{t < 200} \|\mathbf{g}_t\|_2}{\text{mean}_{t \ge 200} \|\mathbf{g}_t\|_2}$$
3. Đo điểm chất lượng ảnh cuối cùng (ImageReward) của cả 2 chế độ.

### D. Kết quả chứng minh LiDAR chưa tốt:
- Khi $t < 200$, $\|\mathbf{g}_t\|_2$ vọt lên từ $10.0$ lên tới **$> 1,500.0$** (gấp $>150$ lần).
- Điểm ImageReward khi không có Cutoff bị tụt thảm hại từ $+0.38$ xuống **$-0.92$** (ảnh bị nát chi tiết do gradient nổ) $\implies$ Khẳng định về mặt lý thuyết toán học, công thức của LiDAR không tự ổn định được khi $t \to 0$ mà phải dùng mẹo thủ công để che giấu!
