# Báo cáo: Cài đặt thực nghiệm — So sánh Diffusion, Score Matching và Flow Matching trên MNIST

**Nguồn:**
- Nhánh `main`: [tamchamchi/diffusion_flow_for_mnist](https://github.com/tamchamchi/diffusion_flow_for_mnist)
- Nhánh `feat/reverse-time-convention`: [tamchamchi/diffusion_flow_for_mnist @ feat/reverse-time-convention](https://github.com/tamchamchi/diffusion_flow_for_mnist/tree/feat/reverse-time-convention)

Repo cài đặt **5 phương pháp sinh ảnh** (2 biến thể flow matching, DDPM, 2 biến thể score matching) trên cùng một **kiến trúc UNet**, cùng một **quy ước thời gian**, và cùng một **bộ giải ODE dòng xác suất (probability-flow ODE)**, để có thể so sánh công bằng. Báo cáo này trình bày công thức cài đặt theo quy ước của nhánh **`feat/reverse-time-convention`** (quy ước hiện hành), đồng thời đối chiếu với quy ước cũ của `main` ở những chỗ khác biệt.

## 1. Công thức trộn nhiễu chung (conditional path)

$$
x_t = \alpha(t)\, x_0 + \sigma(t)\, x_1, \qquad t \in [T_{\min},\, 1-T_{\min}]
$$

| | `main` (quy ước cũ) | `feat/reverse-time-convention` (quy ước hiện hành) |
|---|---|---|
| $x_0$ | dữ liệu (ảnh chữ số sạch) | **nhiễu** $x_0 \sim \mathcal{N}(0, I)$ |
| $x_1$ | nhiễu $x_1 \sim \mathcal{N}(0, I)$ | **dữ liệu** (ảnh chữ số sạch) |
| $t=0$ | gần dữ liệu | **gần nhiễu** |
| $t=1$ | gần nhiễu | **gần dữ liệu** |
| Miền $t$ | $[T_{\min},\, 1]$ | $[T_{\min},\, 1-T_{\min}]$ (cắt ở **cả hai đầu**) |
| Chiều lấy mẫu | tích phân **ngược** ($t: 1\to0$, nhiễu → dữ liệu) | tích phân **thuận** ($t: 0\to1$, nhiễu → dữ liệu) |

Ở quy ước cũ, việc sinh mẫu đòi hỏi giải ODE **lùi theo thời gian** — đúng quy ước "reverse-time" kinh điển của tài liệu diffusion (Song et al., 2021), nơi quá trình thuận đi từ dữ liệu ở $t=0$ đến nhiễu ở $t=1$. Nhánh `feat/reverse-time-convention` **hoán đổi vai trò $x_0 \leftrightarrow x_1$** để bộ lấy mẫu luôn tích phân **thuận theo $t$** — cùng chiều với quy ước Flow Matching/rectified-flow gốc (Lipman et al., 2022), loại bỏ nhu cầu lấy mẫu "ngược thời gian" một cách tường minh. Đây là quy ước dùng xuyên suốt phần còn lại của báo cáo.

Điều kiện biên của $\alpha, \sigma$ theo quy ước hiện hành:

$$
\alpha(T_{\min}) \approx 1,\ \sigma(T_{\min}) \approx 0 \quad\text{(gần nhiễu thuần)} \qquad\qquad \alpha(1-T_{\min}) \approx 0,\ \sigma(1-T_{\min}) \approx 1 \quad\text{(gần dữ liệu thật)}
$$

Sự khác biệt giữa 5 phương pháp nằm ở 3 yếu tố: **(a)** đường dẫn — tuyến tính/OT hay bảo toàn phương sai (VP); **(b)** đại lượng mạng hồi quy — vận tốc $v$, nhiễu $\epsilon$, hay điểm số $\nabla_x\log p_t(x)$; **(c)** trọng số hàm mất mát $w(t)$.

## 2. Bảng 5 phương pháp

| Phương pháp | Họ mô hình | Đường dẫn | Mạng dự đoán | Trọng số loss |
|---|---|---|---|---|
| `fm_ot` | Flow Matching | Linear / Optimal Transport | vận tốc $v$ | đồng nhất ($w=1$) |
| `fm_diffusion` | Flow Matching | Variance-Preserving (VP) | vận tốc $v$ | đồng nhất ($w=1$) |
| `ddpm` | Noise Matching | VP | nhiễu $\epsilon$ | đồng nhất (Ho et al., 2020) |
| `score` | Score Matching | VP | điểm số $s_\theta$ | $\boldsymbol{\alpha(t)^2}$ (Song & Ermon, 2019) |
| `score_flow` | Score Matching | VP | điểm số $s_\theta$ | $\beta(t)$ (trọng số theo likelihood) |

> **Lưu ý về trọng số `score`:** ở `main`, trọng số là $\sigma(t)^2$; ở `feat/reverse-time-convention` là $\alpha(t)^2$. Đây không phải thay đổi thuật toán — trong công thức Song & Ermon (2019), trọng số chuẩn của denoising score matching luôn bằng **bình phương độ lệch chuẩn của thành phần nhiễu tại thời điểm $t$**. Vì thành phần nhiễu giờ được nhân với $\alpha(t)$ (thay vì $\sigma(t)$), trọng số đổi theo tương ứng.

`src/methods/base.py` định nghĩa interface `Method` dùng chung; `src/schedules.py` là nguồn định nghĩa duy nhất cho $\alpha(t), \sigma(t), \beta(t)$.

## 3. Lịch trình đường dẫn (schedules)

### 3.1. Linear / OT — dùng cho `fm_ot`

$$
\alpha(t) = 1-t, \qquad \sigma(t) = t \qquad\Longrightarrow\qquad x_t = (1-t)\,\underbrace{x_0}_{\text{nhiễu}} + t\,\underbrace{x_1}_{\text{dữ liệu}}
$$

Đây là công thức nội suy tuyến tính "nhiễu → dữ liệu" tiêu chuẩn của rectified-flow/optimal-transport flow matching, có đạo hàm không đổi:

$$
\dot\alpha(t) = -1, \qquad \dot\sigma(t) = 1
$$

### 3.2. Variance-Preserving (VP) — dùng cho `fm_diffusion`, `ddpm`, `score`, `score_flow`

Ràng buộc đặc trưng của một schedule VP:

$$
\alpha(t)^2 + \sigma(t)^2 = 1
$$

$\alpha(t)$ là hệ số **nhiễu** (giảm dần $1\to0$), $\sigma(t)$ là hệ số **dữ liệu** (tăng dần $0\to1$). Hệ số nhiễu tức thời:

$$
\beta(t) = -2\,\frac{d}{dt}\log \alpha(t)
$$

## 4. Bộ giải chung: ODE dòng xác suất (Probability-Flow ODE)

Điểm mấu chốt cho phép so sánh công bằng: **mọi phương pháp đều lộ ra một hàm `velocity(x, t)`** — đạo hàm dòng xác suất $dx/dt$ — và chỉ hàm này được bộ lấy mẫu sử dụng, luôn tích phân **thuận**:

$$
\frac{dx}{dt} = v_\theta(x, t), \qquad x(T_{\min}) \sim \mathcal{N}(0,I) \ \longrightarrow\ x(1-T_{\min}) \approx \text{ảnh sinh ra}
$$

**Flow matching (`fm_ot`, `fm_diffusion`)** — mạng dự đoán trực tiếp vận tốc:

$$
v_\theta(x,t) \approx \dot\alpha(t)\, x_0 + \dot\sigma(t)\, x_1
$$

Với đường OT, mục tiêu là hằng số $x_1 - x_0$ (dữ liệu trừ nhiễu) tại mọi $t$.

**Noise/score matching (`ddpm`, `score`, `score_flow`)** — điểm số điều kiện theo dữ liệu $x_1$:

$$
\nabla_{x_t} \log p(x_t \mid x_1) = -\frac{x_t - \sigma(t) x_1}{\alpha(t)^2} = -\frac{x_0}{\alpha(t)}
$$

nên nhiễu $\epsilon_\theta$ (mạng `ddpm` dự đoán) và điểm số $s_\theta$ (mạng `score`, `score_flow` dự đoán) liên hệ:

$$
s_\theta(x,t) = -\frac{\epsilon_\theta(x,t)}{\alpha(t)}
$$

Chuyển từ điểm số sang vận tốc qua công thức dòng xác suất tổng quát cho SDE $dx = f(x,t)\,dt + g(t)\,dw$:

$$
v_\theta(x,t) = f(x,t) - \tfrac{1}{2} g(t)^2\, s_\theta(x,t)
$$

## 5. Hàm mất mát huấn luyện

$$
\mathcal{L}(\theta) = \mathbb{E}_{t,\, x_0,\, x_1}\Big[\, w(t)\, \big\lVert \hat{y}_\theta(x_t, t) - y^\star(x_0, x_1, t) \big\rVert^2 \Big]
$$

| Phương pháp | Mục tiêu $y^\star$ | Trọng số $w(t)$ |
|---|---|---|
| `fm_ot`, `fm_diffusion` | $\dot\alpha(t)x_0 + \dot\sigma(t)x_1$ | $1$ |
| `ddpm` | $x_0$ (nhiễu $\epsilon$) | $1$ (mục tiêu đơn giản hoá, Ho et al. 2020) |
| `score` | $-x_0/\alpha(t)$ | $\alpha(t)^2$ (Song & Ermon, 2019) |
| `score_flow` | $-x_0/\alpha(t)$ | $\beta(t)$ (trọng số theo likelihood, Song et al. 2021) |

Với `score`, khai triển trọng số $\times$ mục tiêu cho thấy loss thực chất tối ưu:

$$
\mathcal{L}_{\text{score}}(\theta) = \mathbb{E}\Big[\big\lVert \alpha(t)\, s_\theta(x_t,t) + x_0 \big\rVert^2\Big]
$$

— dạng không "nổ" khi $\alpha(t)\to0$ (gần phía dữ liệu). Trọng số $w(t)=\beta(t)$ trong `score_flow` biến hàm mất mát thành một **cận trên chặt của negative log-likelihood** (ELBO), khác với $w(t)=\alpha(t)^2$ vốn tối ưu chất lượng mẫu hơn là likelihood.

## 6. Ba chỉ số đánh giá (`src/evaluate_metrics.py`)

Quy trình đánh giá theo Song et al. (2021).

### 6.1. NLL (bits/dim)

Log-likelihood chính xác được tính bằng ODE dòng xác suất mở rộng $(x, \log\text{-det})$ với ước lượng vết Hutchinson (kỹ thuật FFJORD), trên miền đối xứng $[T_{\min}, 1-T_{\min}]$:

$$
\log p_{1-T_{\min}}(x_1) = \log p_{T_{\min}}(x_0) - \int_{T_{\min}}^{1-T_{\min}} \nabla_x \cdot v_\theta(x_t, t)\; dt
$$

Vết của Jacobian được ước lượng không thiên lệch bằng vector ngẫu nhiên Rademacher $\epsilon \sim \{-1,+1\}^d$:

$$
\nabla_x \cdot v_\theta(x_t,t) \approx \epsilon^\top \frac{\partial v_\theta}{\partial x}\, \epsilon
$$

Quy đổi sang đơn vị bits/dim (BPD), với $d = 28\times28 = 784$:

$$
\text{BPD} = \frac{-\log p_{1-T_{\min}}(x_1)}{d \cdot \ln 2} + \text{const}
$$

### 6.2. FID (Fréchet Inception Distance)

$$
\text{FID} = \lVert \mu_r - \mu_g \rVert_2^2 + \text{Tr}\!\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)
$$

trong đó $(\mu_r, \Sigma_r)$ và $(\mu_g, \Sigma_g)$ là trung bình/hiệp phương sai đặc trưng của ảnh thật và ảnh sinh. Repo hỗ trợ **2 bộ trích xuất đặc trưng** (không so sánh chéo được vì khác không gian đặc trưng):

- Inception-v3 (huấn luyện trên ImageNet) — mặc định, dễ so sánh với công bố khác
- MNIST-CNN nhỏ (tự huấn luyện trên MNIST) — phù hợp hơn với ảnh xám 28×28

### 6.3. NFE (Number of Function Evaluations)

Số lần gọi hàm trung bình mà bộ giải thích nghi dung sai (`dopri5`) cần để sinh một mẫu, khi tích phân thuận từ $T_{\min}$ đến $1-T_{\min}$ — đo hiệu quả tính toán của mỗi phương pháp bằng cùng một bộ giải ODE.

## 7. Kết quả thực nghiệm (epoch 350, 2000 mẫu/chỉ số, nhánh `main`)

> Cỡ mẫu 2000 là nhỏ theo chuẩn FID (Heusel et al., 2017 dùng hàng chục nghìn), nên các số liệu mang tính **định hướng**, chưa phải kết quả cuối cùng. Hai bảng dưới dùng hai không gian đặc trưng khác nhau, cột FID **không so sánh chéo được**.

### 7.1. Đặc trưng MNIST-CNN

| Phương pháp | NLL (BPD) ↓ | FID ↓ | NFE trung bình ↓ |
|---|---|---|---|
| `ddpm` | 2.265 | **89.627** | 182.0 |
| `score` | 2.176 | 130.730 | 155.0 |
| `score_flow` | 2.678 | 309.768 | 164.0 |
| `fm_diffusion` | 2.607 | 104.448 | 173.0 |
| `fm_ot` | **1.660** | 96.205 | **122.0** |

### 7.2. Đặc trưng Inception-v3

| Phương pháp | NLL (BPD) ↓ | FID ↓ | NFE trung bình ↓ |
|---|---|---|---|
| `ddpm` | 2.265 | 136.674 | 182.0 |
| `score` | 2.176 | 121.454 | 155.0 |
| `score_flow` | 2.678 | 184.895 | 164.0 |
| `fm_diffusion` | 2.607 | 129.497 | 173.0 |
| `fm_ot` | **1.660** | **119.785** | **122.0** |

**Nhận xét:** `fm_ot` (đường dẫn tuyến tính/OT, dự đoán vận tốc) đạt NLL thấp nhất và NFE ít nhất ở cả hai không gian đặc trưng — phù hợp với lập luận lý thuyết rằng đường dẫn OT tạo ra trường vận tốc "thẳng" hơn, dễ tích phân hơn so với đường dẫn VP dùng trong 4 phương pháp còn lại.

## 8. Vì sao quy ước thời gian mới quan trọng cho việc so sánh công bằng

- **Tính nhất quán khi lấy mẫu:** cả 5 phương pháp giờ dùng chung một chiều tích phân (thuận), loại bỏ nguy cơ sai sót về dấu khi chuyển đổi giữa "reverse-time SDE" của diffusion truyền thống và "forward ODE" của flow matching — vốn dễ nhầm lẫn khi ghép chung một sampler.
- **Cắt cả hai đầu miền $t$** ($[T_{\min}, 1-T_{\min}]$): tránh điểm kỳ dị không chỉ ở phía nhiễu mà cả ở phía dữ liệu — nơi $\sigma(t)\to1, \alpha(t)\to0$ cũng có thể gây bất ổn số học cho các phương pháp score-based.
- **Trọng số `score` đổi theo đúng lý thuyết:** đảm bảo denoising score matching vẫn đúng công thức gốc (Song & Ermon, 2019) sau khi hoán đổi vai trò $x_0/x_1$.

## 9. Tài liệu tham khảo

- Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising diffusion probabilistic models*. NeurIPS 33, 6840–6851. <https://arxiv.org/abs/2006.11239>
- Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). *Score-based generative modeling through stochastic differential equations*. arXiv:2011.13456.
- Song, Y., & Ermon, S. (2019). *Generative modeling by estimating gradients of the data distribution*. NeurIPS 32.
- Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2022). *Flow matching for generative modeling*. arXiv:2210.02747.
- Luo, C. (2022). *Understanding diffusion models: A unified perspective*. arXiv:2208.11970.
- Holderrieth, P., & Erives, E. (2025). *An introduction to flow matching and diffusion models*. arXiv:2506.02070.

---
*Ghi chú phương pháp luận:* Bảng kết quả (Mục 7) lấy trực tiếp từ README nhánh `main`. Các công thức về $\dot\alpha, \dot\sigma$, hệ số $\beta(t)$, và quan hệ $\epsilon_\theta \leftrightarrow s_\theta \leftrightarrow v_\theta$ trong Mục 3–5 được suy ra từ mô tả README của nhánh `feat/reverse-time-convention` kết hợp công thức chuẩn trong các tài liệu được trích dẫn. Nếu cần đối chiếu chính xác tuyệt đối (hằng số cụ thể trong $\beta(t)$, dạng VP schedule — cosine, linear-beta,...), nên tham chiếu trực tiếp `src/schedules.py` và `src/methods/*.py` trên nhánh tương ứng.