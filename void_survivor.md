# Tài Liệu Thiết Kế Hệ Thống Reinforcement Learning: Né Đạn & Diệt Boss

Tài liệu này đặc tả kiến trúc hệ thống Học tăng cường (Reinforcement Learning - RL) áp dụng cho tựa game bắn súng né đạn (Bullet Hell). Hệ thống sử dụng thuật toán **MaskablePPO** kết hợp với cơ chế **Robust Auto-Curriculum Learning (Tự động hóa chương trình học)**, áp dụng các kỹ thuật nâng cao như **Stage Mixing** và **Automatic Rollback** để giải quyết triệt để bài toán quên kiến thức cũ (Catastrophic Forgetting) và sập mô hình (Policy Collapse).

---

## I. Định Nghĩa Thành Phần RL (MDP Framework)

Để đảm bảo Agent học một cách nhất quán, không gian Trạng thái (State), Hành động (Action) và hàm Thưởng (Reward) được **giữ cố định cấu trúc** xuyên suốt tất cả các giai đoạn huấn luyện.

### 1. Không gian Trạng thái (State Space)

Toàn bộ tọa độ được tính toán **tương đối dựa trên vị trí Agent** (lấy vị trí Agent làm gốc tọa độ $(0,0)$) và ép về khoảng $[-1, 1]$ hoặc $[0, 1]$ để tối ưu hóa tốc độ hội tụ của mạng Neural.

* **Ký hiệu:** * Kích thước màn hình: $(W, H)$
  * Khoảng cách chéo tối đa: $D_{max} = \sqrt{W^2 + H^2}$
  * Vận tốc đạn tối đa trong cấu hình game: $V_{max}$

Vector State là một mảng **45 chiều** cố định bao gồm:

| Nhóm Tính Năng | Số Chiều | Công Thức / Định Dạng | Ý Nghĩa Giải Thuật |
| :--- | :---: | :--- | :--- |
| **Agent Status** | 2 | `[HP_current / HP_max, Cooldown / Cooldown_max]` | Trạng thái sinh tồn và khả năng xả đạn của Agent. |
| **Boss Status** | 3 | `[(X_boss - X_agent)/W, (Y_boss - Y_agent)/H, HP_boss / HP_boss_max]` | Vị trí tương đối của Boss so với Agent và lượng máu còn lại. |
| **Bullet Radar** | 40 | 10 viên đạn gần Agent nhất. Mỗi viên: `[(X_b - X_a)/W, (Y_b - Y_a)/H, Vx_b/V_max, Vy_b/V_max]` | Giúp Agent nhận biết sớm quỹ đạo di chuyển của các mối đe dọa cận kề. |

*Lưu ý: Nếu số lượng đạn trên màn hình ít hơn 10, toàn bộ các ô trống còn lại trong Bullet Radar sẽ được điền giá trị 0.*

### 2. Không gian Hành động (Action Space) & Action Masking

Sử dụng **Discrete Action Space** gồm 9 hành động kết hợp giữa di chuyển 4 hướng và trạng thái bắn:

* `0`: Đứng yên
* `1`: Di chuyển Lên | `2`: Di chuyển Xuống | `3`: Di chuyển Trái | `4`: Di chuyển Phải
* `5`: Lên + Bắn | `6`: Xuống + Bắn | `7`: Trái + Bắn | `8`: Phải + Bắn

> 💡 **Cơ chế Action Masking:** Khi `Cooldown > 0` (súng đang nạp đạn), môi trường game sẽ sinh ra một mặt nạ hành động (action mask): `[True, True, True, True, True, False, False, False, False]`. Thuật toán `MaskablePPO` sẽ chủ động khóa các hành động từ 5 đến 8, ngăn Agent lãng phí tài nguyên tính toán vào việc thử nghiệm hành động bắn vô nghĩa.

### 3. Hàm Thưởng (Reward Function)

Hàm reward tổng quát tại mỗi step được cấu trúc theo công thức:

$$Reward = R_{survival} + R_{offensive} + R_{terminal}$$

#### Nhánh Sinh Tồn ($R_{survival}$):
* **Frame Reward:** $+0.005$ cho mỗi frame sống sót (giữ mức nhỏ để tránh Agent cố tình câu giờ).
* **Phạt Trúng Đạn:** $-1.0$ ngay khi Agent bị trừ HP.
* **Thưởng Khoảng Cách An Toàn:** $+0.01 \times \left( \frac{\text{Distance}(Agent, Bullet_{closest})}{D_{max}} \right)$ (Đã chuẩn hóa để tránh việc Agent quá nhát gan, ưu tiên trốn ở góc xa thay vì lao vào diệt Boss).

#### Nhánh Tấn Công ($R_{offensive}$):
* **Thưởng Sát Thương:** $+0.2 \times (\text{Sát thương gây ra})$.
* **Phạt Bắn Hụt:** $-0.05$ nếu Agent chọn hành động Bắn nhưng đạn bay ra ngoài màn hình mà không trúng Boss (Hạn chế việc AI spam nút bắn bừa bãi).

#### Nhánh Kết Thúc ($R_{terminal}$):
* **Thắng Trận (Boss HP = 0):** $+50.0$
* **Thua Trận (Agent HP = 0):** $-20.0$

---

## II. Kiến Trúc Curriculum Learning & Chiến Lược Stage Mixing

Độ khó được tăng tiến tuyến tính thông qua việc cấu hình thực thể Boss trong môi trường, trong khi cấu trúc hàm Reward hoàn toàn được giữ nguyên.

### 1. Phân Chia Giai Đoạn (Stages)

* **Stage 1 (Tập bắn):** Boss đứng yên, HP thấp ($100$). Boss không bắn đạn. Mục tiêu: Agent học cơ chế di chuyển lại gần Boss và bắn để lấy reward dương.
* **Stage 2 (Né cơ bản):** Boss đứng yên. Boss bắn đạn thẳng, tần suất chậm (2 giây/viên). Mục tiêu: Agent học cách di chuyển qua lại để né đạn đan xen với việc bắn trả.
* **Stage 3 (Mục tiêu di động):** Boss di chuyển qua lại. Tần suất bắn tăng, đạn bay nhanh hơn. Mục tiêu: Agent học cách vừa đuổi theo mục tiêu di động vừa luồn lách né đạn.
* **Stage 4 (Full Phase):** Boss di chuyển thông minh, bắn đạn chùm (Spread) hoặc đạn đuổi (Homing). Mục tiêu: Tối ưu hóa Policy đến mức thượng thừa.

### 2. Chiến Lược Trộn Giai Đoạn (Stage Mixing)

Để giải quyết triệt để hiện tượng *Recency Bias* (chỉ nhớ môi trường gần nhất) gây ra lỗi quên kiến thức cũ, tại mỗi lượt `reset()` môi trường để bắt đầu một trận đấu mới, cấu hình Stage của trận đó sẽ được chọn ngẫu nhiên theo tỷ lệ:

* **70% xác suất:** Khởi chạy `Current_Stage` hiện tại.
* **20% xác suất:** Khởi chạy `Current_Stage - 1` (Stage liền kề trước đó để ôn bài).
* **10% xác suất:** Khởi chạy ngẫu nhiên một trong các Stage cũ hơn tính từ đầu game.

---

## III. Quy Trình Huấn Luyện & Cơ Chế Đảm Bảo Độ Ổn Định

### 1. Đánh Giá Độc Lập (Decoupled Evaluation)
Tách biệt hoàn toàn quá trình Huấn luyện (Training) và Đánh giá (Evaluation). Định kỳ mỗi $50,000$ steps huấn luyện, hệ thống sẽ chạy một luồng Evaluation độc lập gồm 20 trận test cho **từng Stage một** (từ Stage 1 đến Stage hiện tại) để ghi nhận Win-rate chuẩn xác sang TensorBoard.
* Điều kiện nâng Stage: Win-rate của Stage hiện tại đạt $> 80\%$, đồng thời Win-rate các Stage cũ không bị sụt giảm quá $5\%$.

### 2. Hệ Thống Checkpoint & Tự Động Rollback
Khi môi trường tăng độ khó đột ngột, gradient của mạng dễ bị bùng nổ dẫn đến hỏng toàn bộ trọng số (Policy Collapse). 
* Hệ thống liên tục lưu lại file `best_model.zip` dựa trên tổng điểm Win-rate trung bình cao nhất tại luồng Evaluation.
* Nếu sau khi nâng Stage, Win-rate của Stage hiện tại tụt xuống dưới $15\%$ liên tục trong $3$ kỳ Evaluation kế tiếp, hệ thống sẽ tự động thực hiện **Rollback**: Tải lại trọng số khỏe mạnh từ `best_model.zip` gần nhất, đồng thời hạ `Current_Stage` xuống 1 cấp.

### 3. Điều Khiển Hệ Số Entropy Động (Dynamic Entropy Schedule)
Hệ số Entropy (`ent_coef`) quyết định mức độ khám phá (Exploration) của Agent. 
* Tại Stage 1: Thiết lập `ent_coef = 0.05` để Agent tích cực di chuyển tìm mục tiêu.
* Mỗi khi hệ thống nâng lên một Stage mới: Tự động cộng thêm `+0.02` vào `ent_coef` hiện tại trong vòng $100,000$ steps đầu tiên để kích thích Agent tái khám phá, tìm giải pháp cho các làn đạn mới thay vì lười biếng áp dụng Policy cũ. Sau đó giảm dần về mức nền `0.01`.

---

## IV. Cấu Trúc Mã Nguồn Minh Họa (Python & SB3-Contrib)

### 1. Triển Khai Môi Trường Custom (`environment.py`)

```python
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class BulletHellCurriculumEnv(gym.Env):
    def __init__(self):
        super(BulletHellCurriculumEnv, self).__init__()
        
        # 9 Discrete Actions tương ứng với thiết kế
        self.action_space = spaces.Discrete(9)
        
        # 45 Features tương đối đã được chuẩn hóa về khoảng [-1, 1]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(45,), dtype=np.float32)
        
        # Biến quản lý tiến trình Curriculum
        self.current_stage = 1       # Stage cao nhất hiện tại mà Agent đạt tới
        self.active_match_stage = 1  # Stage thực tế áp dụng cho trận đấu hiện tại

    def set_stage(self, stage):
        self.current_stage = stage

    def action_masks(self):
        """Trả về mảng boolean khóa hành động bắn khi súng đang trong trạng thái hồi chiêu"""
        mask = np.ones(9, dtype=bool)
        if self.game.agent.cooldown > 0:
            mask[5:9] = False  # Khóa các hành động tích hợp bắn [5, 6, 7, 8]
        return mask

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Thực hiện chiến lược Stage Mixing (70% / 20% / 10%)
        rand = np.random.rand()
        if rand < 0.70 or self.current_stage == 1:
            self.active_match_stage = self.current_stage
        elif rand < 0.90:
            self.active_match_stage = max(1, self.current_stage - 1)
        else:
            self.active_match_stage = np.random.randint(1, max(2, self.current_stage))
            
        # Khởi tạo cấu hình Boss/Đạn dựa trên biến self.active_match_stage
        self.game.init_match(stage=self.active_match_stage)
        
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        # Thực thi logic game dựa trên action nhận được từ Agent
        self.game.update(action)
        
        obs = self._get_obs()
        reward = self._calculate_reward()
        terminated = self.game.is_over()
        truncated = False
        
        info = {
            "is_win": self.game.boss.hp <= 0,
            "match_stage": self.active_match_stage
        }
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # [Tính toán logic nội bộ]: Chuyển đổi tọa độ sang tương đối và chuẩn hóa về [-1, 1]
        obs_vector = np.zeros(45, dtype=np.float32)
        return obs_vector

    def _calculate_reward(self):
        # [Tính toán logic nội bộ]: Hiện thực hóa hàm reward động tại mục I.3
        return 0.0


2. Triển Khai Vòng Lặp Huấn Luyện (train.py)

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env

def make_env():
    env = BulletHellCurriculumEnv()
    # Bọc môi trường bằng ActionMasker để kích hoạt tính năng đóng/mở Action động
    env = ActionMasker(env, lambda e: e.action_masks())
    return env

def evaluate_all_stages(model):
    """
    Hàm đánh giá độc lập hiệu năng của Agent trên từng Stage cụ thể.
    Trả về: (win_rate_current_stage, is_old_stage_corrupted)
    """
    # [Hiện thực hóa logic chạy thử nghiệm độc lập tại đây]
    return 0.85, False

if __name__ == "__main__":
    # Song song hóa môi trường (Vectorized Environment) để tăng tốc độ huấn luyện
    num_envs = 8
    vec_env = make_vec_env(make_env, n_envs=num_envs)

    # Khởi tạo mô hình MaskablePPO
    model = MaskablePPO(
        "MlpPolicy", 
        vec_env, 
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.05,  # Hệ số Entropy cao ở giai đoạn đầu trận để tăng khám phá
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )

    current_global_stage = 1
    total_iterations = 100

    for iteration in range(total_iterations):
        # Huấn luyện mô hình định kỳ theo từng block step
        model.learn(total_timesteps=50000, reset_num_timesteps=False)
        
        # Chạy luồng Evaluation độc lập tách biệt khỏi tiến trình Train
        win_rate, corrupted = evaluate_all_stages(model) 
        
        # Kiểm tra điều kiện nâng cấp Stage
        if win_rate > 0.80 and not corrupted and current_global_stage < 4:
            current_global_stage += 1
            # Đồng bộ cấu hình Stage mới xuống toàn bộ các môi trường song song
            vec_env.env_method("set_stage", current_global_stage)
            
            # Lưu lại Checkpoint an toàn
            model.save(f"best_model_stage_{current_global_stage}")
            
            # Kích hoạt Dynamic Entropy Schedule: Tăng nhẹ để ép Agent học kỹ năng mới
            model.ent_coef = 0.04 
            print(f"--- TIẾN TIẾN GIAI ĐOẠN: LÊN STAGE {current_global_stage} ---")
            
        # Kiểm tra điều kiện sập mô hình để thực hiện Rollback
        elif win_rate < 0.15 and current_global_stage > 1:
            print("!!! PHÁT HIỆN SẬP POLICY - KÍCH HOẠT QUY TRÌNH ROLLBACK !!!")
            # Tải lại checkpoint ổn định gần nhất của stage trước đó
            model = MaskablePPO.load(f"best_model_stage_{current_global_stage}", env=vec_env)
            current_global_stage -= 1
            vec_env.env_method("set_stage", current_global_stage)


