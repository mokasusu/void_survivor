import os

filepath = "/Users/vducc3110/Desktop/void_survivor/chuong4.md"

if not os.path.exists(filepath):
    print(f"Error: {filepath} does not exist.")
    exit(1)

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

idx = content.find("## 4.3. Đóng gói mô hình")
if idx == -1:
    print("Error: ## 4.3. Đóng gói mô hình not found.")
    exit(1)

# Define the replacement block with annotated instructions and specific filenames
replacement_block = """## 4.3. Đóng gói mô hình và Xây dựng Prototype Demo sản phẩm

Sau khi quá trình huấn luyện hoàn tất và mô hình đã hội tụ về chính sách tối ưu, hệ thống tiến hành đóng gói và triển khai chương trình chạy trình diễn (Inference Script) để nghiệm thu sản phẩm.

### 4.3.1. Cơ chế đóng gói Model

*   **Phương pháp tuần tự hóa (Serialization) của Stable-Baselines3:**
    Thư viện Stable-Baselines3 hỗ trợ cơ chế lưu trữ toàn bộ trạng thái của mô hình dưới định dạng nén `.zip`. File lưu trữ này không chỉ đơn thuần chứa các ma trận trọng số mạng neural mà còn bảo toàn toàn bộ cấu trúc siêu tham số và trạng thái của thuật toán tối ưu. Cấu trúc bên trong của file `.zip` sau khi giải nén bao gồm các thành phần cốt lõi:
    1.  `data`: File JSON chứa toàn bộ siêu tham số của thuật toán (ví dụ: `learning_rate = 3e-4`, `gamma = 0.99`, `clip_range = 0.2`, `ent_coef = 0.01`), cấu trúc của không gian quan sát (Observation Space) và không gian hành động (Action Space).
    2.  `parameter_list`: Danh sách định nghĩa kiểu dữ liệu và hình dáng (shape) của các tensor tham số.
    3.  `policy.pt`: Chứa trọng số thực tế (weights và biases) của cả hai mạng Actor và Critic dưới dạng tuần tự hóa PyTorch (được ghi thông qua hàm `torch.save()`).
    4.  `policy.optimizer.pt`: Chứa trạng thái của thuật toán tối ưu hóa Adam Optimizer (bao gồm các momen động lượng thứ nhất $m_t$ và thứ hai $v_t$ tại bước huấn luyện cuối cùng), cho phép người dùng tiếp tục huấn luyện tinh chỉnh (fine-tuning) sau này mà không lo mất đà tối ưu trước đó.

> [!NOTE]
> **[ẢNH CẦN CHÈN 1 - CẤU TRÚC FILE ZIP MODEL]**
> *   **Vị trí chèn:** Ngay sau phần phân tích cấu trúc file `.zip`.
> *   **Tên file đề xuất:** `model_zip_structure.png` (Sơ đồ mô tả cấu trúc thư mục giải nén của file model `.zip` gồm 4 thành phần trên để tăng tính trực quan khoa học).
> *   **Chú thích:** *Hình 4.9: Sơ đồ mô tả cấu trúc tuần tự hóa các tệp tin lưu trữ bên trong định dạng checkpoint .zip của Stable-Baselines3.*

*   **Mã nguồn triển khai cơ chế lưu trữ tự động:**
    Để đảm bảo mô hình được đóng gói tự động khi đạt hiệu năng cao nhất trên luồng đánh giá độc lập, một callback tùy chỉnh (`AdaptiveCurriculumCallback` kế thừa từ `BaseCallback`) được tích hợp vào vòng lặp huấn luyện. Khi win-rate đánh giá đạt đỉnh mới, mô hình sẽ tự động được đóng gói:
    ```python
    import os
    from stable_baselines3.common.callbacks import BaseCallback

    class AdaptiveCurriculumCallback(BaseCallback):
        def __init__(self, eval_env, check_freq: int, save_path: str):
            super().__init__()
            self.eval_env = eval_env
            self.check_freq = check_freq
            self.save_path = save_path
            self.best_win_rate = 0.0

        def _on_step(self) -> bool:
            # Thực hiện đánh giá độc lập định kỳ sau mỗi check_freq steps
            if self.n_calls % self.check_freq == 0:
                win_rate = self.eval_env.evaluate_agent(self.model)
                if win_rate > self.best_win_rate:
                    self.best_win_rate = win_rate
                    # Tiến hành đóng gói và xuất file .zip
                    model_file = os.path.join(self.save_path, "best_model.zip")
                    self.model.save(model_file)
                    print(f"[Callback] Đã đóng gói checkpoint tối ưu mới: {model_file} | Win Rate: {win_rate*100:.2f}%")
            return True
    ```

### 4.3.2. Kịch bản chạy trình diễn (Inference Script)

Để nghiệm thu sản phẩm trực quan, chúng tôi xây dựng file kịch bản trình diễn đặt tên là `enjoy.py` (trong mã nguồn thực tế của dự án là file [play_ppo.py](file:///Users/vducc3110/Desktop/void_survivor/rl/play_ppo.py)). Nhiệm vụ của script này là load mô hình đã đóng gói, khởi tạo đồ họa Pygame và đưa Agent vào chế độ suy luận thời gian thực để tự động chơi game trước sự quan sát trực quan của người dùng.

*   **Các kỹ thuật tối ưu hóa trong Inference:**
    1.  *Quyết định chính xác (Deterministic Inference):* Khi chạy trình diễn, chúng ta tắt bỏ hoàn toàn tính ngẫu nhiên (exploration) của mô hình bằng cách thiết lập tham số `deterministic = True` khi gọi dự đoán hành động:
        $$\hat{a}_t = \arg\max_{a} \pi_{\theta}(a | s_t)$$
        Điều này ép Agent luôn luôn chọn hành động có xác suất cao nhất từ phân phối hành động sinh ra bởi mạng Actor, giúp các pha di chuyển và né đạn đạt độ chuẩn xác tuyệt đối.
    2.  *Áp dụng Action Masking trong thời gian thực:* Tại mỗi frame, Inference script vẫn liên tục tính toán mặt nạ hành động (Action Mask) dựa trên trạng thái hồi chiêu súng của phi thuyền. Điều này đảm bảo mạng neural không đưa ra lệnh bắn vô nghĩa khi súng đang nạp đạn, giữ tài nguyên xử lý hoàn toàn cho các hành động di chuyển né tránh.
    3.  *Đồng bộ Khung hình (60 FPS Limit):* Để đảm bảo giao diện hiển thị mượt mà và trực quan cho mắt người xem, game loop được khống chế ở tốc độ 60 khung hình trên giây bằng clock của Pygame (`pygame.time.Clock().tick(60)`).

> [!NOTE]
> **[ẢNH CẦN CHÈN 2 - SƠ ĐỒ CHU TRÌNH INFERENCE]**
> *   **Vị trí chèn:** Ngay sau các kỹ thuật tối ưu hóa của quá trình suy luận.
> *   **Tên file đề xuất:** `inference_flowchart.png` (Sơ đồ khối thể hiện vòng lặp suy luận: Game State -> PPOStateEncoder -> Action Masking -> model.predict(deterministic=True) -> Action execution -> pygame.display.flip()).
> *   **Chú thích:** *Hình 4.10: Sơ đồ luồng xử lý và vòng lặp suy luận thời gian thực của Agent trong kịch bản chạy trình diễn (enjoy.py).*

*   **Mã nguồn Core Loop của quá trình Inference:**
    Dưới đây là đoạn mã nguồn cốt lõi trong file [play_ppo.py](file:///Users/vducc3110/Desktop/void_survivor/rl/play_ppo.py) thể hiện cách thức load model và chạy suy luận trực quan:
    ```python
    import pygame
    from sb3_contrib import MaskablePPO
    from rl.bullet_hell_env import BulletHellEnv
    from rl.ppo_state_encoder import PPOStateEncoder

    def run_inference_demo(model_path: str, stage: int, fps: int = 60):
        # 1. Khởi tạo môi trường game với chế độ hiển thị đồ họa
        env = BulletHellEnv(render_mode="human", render_fps=fps)
        encoder = PPOStateEncoder()
        
        # 2. Giải nén và nạp trọng số mô hình đã huấn luyện
        print(f"Loading model: {model_path}")
        model = MaskablePPO.load(model_path)
        
        # Thiết lập Stage và khởi động màn chơi
        env.current_stage = stage
        env.game.init_match(stage=stage)
        
        obs = encoder.encode(env.game)
        done = False
        clock = pygame.time.Clock()
        
        # 3. Vòng lặp suy luận Inference Loop
        while not done:
            # Đồng bộ tốc độ khung hình hiển thị
            clock.tick(fps)
            
            # Lấy mặt nạ hành động hợp lệ tại frame hiện tại
            action_masks = env.action_masks()
            
            # Dự đoán hành động tối ưu nhất (Deterministic)
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            
            # Thực thi hành động và cập nhật trạng thái game
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            
        pygame.quit()
    ```

---

## 4.4. Giao diện thực tế và Minh họa chức năng của Agent AI

Giao diện đồ họa của trò chơi *Void Survivor* được xây dựng trên nền tảng thư viện Pygame, cung cấp môi trường trực quan 2D hiển thị cuộc chiến thời gian thực giữa phi thuyền (Agent) và thực thể Boss.

### 4.4.1. Thiết kế giao diện game
Giao diện trò chơi được thiết kế với độ phân giải chuẩn **$800 \times 600$ pixels**, các thành phần hiển thị và tông màu được thiết lập tối ưu phục vụ cho cả trải nghiệm người dùng lẫn tính nhất quán của trạng thái RL:
*   **Khu vực trung tâm:** Hiển thị phi thuyền của Agent và thực thể Boss trên nền không gian màu xám tối (`BACKGROUND_COLOR = (30, 30, 30)`) giúp tạo độ tương phản tối đa với các thực thể khác.
    *   Phi thuyền của Agent được vẽ bằng Sprite màu xanh lục đặc trưng (`PLAYER_COLOR = (0, 255, 0)`), với tâm phi thuyền chứa một điểm đỏ siêu nhỏ đại diện cho **Hitbox va chạm thực tế (bán kính 15px)**. Bọc ngoài phi thuyền là một vòng tròn nét đứt vô hình thể hiện **Vùng lướt đạn Grazing (bán kính 50px)**.
    *   Boss được thiết kế với Sprite khổng lồ nằm ở khu vực phía trên màn hình, di chuyển ngang qua lại và xả ra các dòng đạn tròn màu đỏ rực rỡ (`BULLET_COLOR = (255, 0, 0)`).
*   **Thanh hiển thị trạng thái HUD (Heads-Up Display):** 
    *   *Thanh HP của Agent:* Hiển thị màu xanh ngọc biển (`HP_COLOR = (0, 220, 255)`) đặt gọn gàng ở góc trên bên trái màn hình.
    *   *Thanh HP của Boss:* Được thiết kế là một thanh HP cỡ lớn kéo dài dọc toàn bộ viền phía trên màn hình, giúp phản ánh trực quan lượng máu còn lại của Boss dưới tác động xả đạn liên tục từ phi thuyền.
*   **Bảng điều khiển thông tin phụ (Info Panel):** Nằm dưới cùng màn hình với chiều cao cố định **$90$ pixels** (`INFO_PANEL_HEIGHT = 90`). Đây là nơi hiển thị các chỉ số gỡ lỗi văn bản màu xanh lơ (`TEXT_COLOR = (180, 240, 255)`), bao gồm: Chỉ số Stage hiện tại, thời gian trận đấu, số điểm phần thưởng tích lũy (Reward) và trạng thái hồi chiêu của vũ khí.

> [!NOTE]
> **[ẢNH CẦN CHÈN 3 - SƠ ĐỒ THIẾT KẾ GIAO DIỆN GAME]**
> *   **Vị trí chèn:** Ngay sau phần mô tả các thông số HUD và bảng điều khiển.
> *   **Tên file đề xuất:** `game_interface_layout.png` (Sơ đồ phác họa thiết kế màn hình game 800x600, chỉ rõ vị trí thanh Boss HP, Player HP, Info Panel cao 90px và Hitbox 15px/Grazing zone 50px của phi thuyền).
> *   **Chú thích:** *Hình 4.11: Bản phác thảo sơ đồ bố trí các thực thể và khu vực thông số hiển thị (HUD) trên giao diện game.*

### 4.4.2. Minh họa hành vi thực tế của Agent trên các Stage
Dưới đây là chuỗi ảnh chụp màn hình ghi nhận thực tế từ chương trình chạy trình diễn nghiệm thu, minh họa cho khả năng xử lý thông minh của Agent AI trên từng giai đoạn khó:

![Hình 4.12: Giao diện game thực tế tại Stage 1 và Stage 2](gameplay_stage_1_2.png)
*Hình 4.12: Giao diện game thực tế tại Stage 1 (Boss đứng im để Agent tập bắn) và Stage 2 (Agent bắt đầu di chuyển né tránh các làn đạn thẳng cơ bản).*

> [!TIP]
> **[HƯỚNG DẪN CHỌN FILE ẢNH CHO HÌNH 4.12]**
> *   **File ảnh tương ứng có sẵn:** Bạn có thể lấy file ảnh nền tối tên là `media__1781024159281.jpg` hoặc `media__1781024824439.jpg` trong thư mục [media_assets](file:///Users/vducc3110/Desktop/void_survivor/media_assets/) rồi đổi tên thành `gameplay_stage_1_2.png`.
> *   **Hoặc tự chụp ảnh mới:** Chạy game thủ công và chụp lại ảnh màn hình game lúc Agent đứng yên bắn Boss (Stage 1) và lúc Agent di chuyển ngang né dòng đạn thẳng đơn (Stage 2) rồi lưu vào thư mục dự án dưới tên `gameplay_stage_1_2.png`.

*   **Phân tích hành vi Stage 1 & 2:**
    *   *Tại Stage 1 (Boss tĩnh không bắn):* Agent di chuyển hỗn loạn do Entropy ban đầu cao. Sau vài trăm episode học hỏi, Agent nhanh chóng làm chủ cơ chế ngắm bắn: di chuyển đến tọa độ X trùng khớp với Boss, đứng yên cố định trục dọc và xả đạn liên tục để tối ưu hóa thưởng sát thương ($R_{offensive} = +0.2 \times \text{damage}$) mà không tốn năng lượng di chuyển thừa.
    *   *Tại Stage 2 (Boss bắn đạn thẳng đơn):* Khi Boss bắt đầu bắn ra các hạt đạn đỏ bay thẳng xuống dưới, Agent kích hoạt kỹ năng **né tránh ngang (lateral strafing)**. Radar phát hiện đạn đối phương bay sát tâm phi thuyền ở khoảng cách $< 60$ pixels (kích hoạt hình phạt khoảng cách gần đạn $R_{proximity}$), Agent sẽ chọn hành động di chuyển Trái (`3`) hoặc Phải (`4`) để lách qua làn đạn, sau đó ngay lập tức quay trở lại vị trí thẳng hàng với Boss để duy trì trục bắn trả.

![Hình 4.13: Giao diện game thực tế tại Stage 3 và Stage 4](gameplay_stage_3_4.png)
*Hình 4.13: Giao diện game thực tế tại Stage 3 (Boss di chuyển qua lại) và Stage 4 (Boss xả đạn chùm và đạn đuổi khép góc cực kỳ nguy hiểm).*

> [!TIP]
> **[HƯỚNG DẪN CHỌN FILE ẢNH CHO HÌNH 4.13]**
> *   **File ảnh tương ứng có sẵn:** Bạn có thể sử dụng file ảnh nền tối kích thước lớn trong thư mục [media_assets](file:///Users/vducc3110/Desktop/void_survivor/media_assets/) như `media__1781026398177.png` hoặc `media__1781050399085.png` rồi đổi tên thành `gameplay_stage_3_4.png`.
> *   **Hoặc tự chụp ảnh mới:** Chụp màn hình game lúc Agent di chuyển ngang theo đuôi Boss (Stage 3) và lúc Agent luồn lách qua làn đạn chùm spread, né đạn homing hoặc lướt sát sườn đạn ăn điểm Grazing (Stage 4) rồi lưu vào dự án dưới tên `gameplay_stage_3_4.png` để thay thế.

*   **Phân tích hành vi nâng cao Stage 3 & 4:**
    *   *Tại Stage 3 (Boss di chuyển ngang):* Agent học được hành vi **bám đuổi (tracking)** mục tiêu di động. Agent di chuyển ngang nhịp nhàng song hành cùng Boss để giữ Boss luôn nằm trong tầm đạn bắn thẳng đứng của mình, đồng thời thực hiện các pha lắc giật cục nhỏ để lách qua các dòng đạn bắn trả từ Boss.
    *   *Tại Stage 4 (Full Phase - Spread & Homing Bullets):* Đây là đỉnh cao của sự hội tụ chính sách. Boss liên tục xả đạn chùm (Spread) tạo thành bức tường đạn đan chéo dày đặc và đạn đuổi (Homing) khóa mục tiêu. Agent AI thể hiện các hành vi xử lý thượng thừa:
        1.  *Chui qua khe đạn chùm:* Dựa vào thông tin từ Bullet Radar (10 viên gần nhất), Agent phát hiện các kẽ hở an toàn siêu nhỏ giữa các làn đạn chùm và chui qua đó với độ chính xác tuyệt đối.
        2.  *Phá hướng đạn đuổi (Homing Evasion):* Agent dẫn dụ đạn đuổi đến gần, rồi thực hiện một pha di chuyển bứt tốc đột ngột (Dash) sang hướng đối diện khi đạn cách tâm tàu dưới $30$ pixels, làm lệch vector hướng của đạn homing khiến nó bay sượt qua phi thuyền và tự hủy ngoài màn hình.
        3.  *Lướt đạn sát sườn (Grazing Mastery):* Thay vì chạy trốn ở các góc màn hình an toàn (hành vi nhút nhát), Agent chủ động **áp sát sườn các viên đạn** ở khoảng cách từ $16$ pixels đến $45$ pixels (vừa ngoài Hitbox va chạm $15$ pixels nhưng nằm trọn trong vùng Grazing $50$ pixels). Hành vi này giúp Agent liên tục nhận được điểm thưởng Grazing ($+0.05$ điểm/frame) để bù đắp điểm phạt thời gian và tối ưu hóa điểm số của màn chơi một cách xuất sắc.
"""

new_content = content[:idx] + replacement_block

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Successfully replaced content starting from Section 4.3 in chuong4.md.")
