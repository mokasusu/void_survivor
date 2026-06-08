"""
auto_curriculum_env.py — AutomatedCurriculumWrapper cho 6-Stage Bullet Hell.

Tích hợp theo đặc tả kỹ thuật:
  - Phase-based Stage Distribution (5 Phase, tự động thăng)
  - Rolling Window Win Rate (200 episodes / stage)
  - Potential-based Reward Scaling (×1.0 → ×2.25 tăng dần theo Stage)
  - Auto-Backup khi thăng Phase (gọi model_saver_callback)
  - Duy trì rehearsal tối thiểu 10% cho Stage cũ (chống Catastrophic Forgetting)

Cách dùng:
    from rl.auto_curriculum_env import AutomatedCurriculumWrapper
    curriculum_env = AutomatedCurriculumWrapper(raw_env)
    # Sau khi model được khởi tạo, AdaptiveHyperparameterCallback sẽ
    # gắn model_saver_callback vào wrapper này.
"""

import random
import numpy as np
import gymnasium as gym
from collections import deque


class AutomatedCurriculumWrapper(gym.Wrapper):
    """
    Wrapper tự động hóa Curriculum Learning cho 6 Stage theo 5 Phase tiến độ.

    Phase 1 → 2 → 3 → 4 → 5 dựa trên Rolling Window Win Rate của từng Stage.
    Mỗi Phase có phân phối Stage tối ưu để cân bằng "học mới" và "ôn cũ".
    """

    def __init__(self, env, model_saver_callback=None):
        """
        :param env: BulletHellEnv gốc (chưa bọc ActionMasker)
        :param model_saver_callback: Hàm (filename: str) -> None để lưu checkpoint.
                                     Được AdaptiveHyperparameterCallback gắn vào sau.
        """
        super().__init__(env)
        self.train_phase: int = 1
        self.current_stage: int = 1
        self.model_saver_callback = model_saver_callback

        # Rolling Window 200 episodes cho từng Stage (1-6)
        self.window_size: int = 200
        self.stage_history: dict[int, deque] = {
            i: deque(maxlen=self.window_size) for i in range(1, 7)
        }

        # Ma trận phân phối Stage theo Phase
        # Thiết kế theo nguyên tắc Cold-Start Safety:
        #
        #   Phase 1 (Cold Start): 85% Stage 1 — model phải nắm vững nền tảng trước.
        #     Chỉ preview nhẹ Stage 2(10%) và Stage 3(5%) để tránh gradient shock.
        #     Nếu bắt đầu bằng 45% Stage 2 ngay từ đầu → cold-start model không hội tụ được.
        #
        #   Phase 2: Trọng tâm Stage 3-4, giữ rehearsal tối thiểu Stage 1-2
        #   Phase 3: Tập trung Stage 4-5 (45%+35%), giảm Stage 1-2 xuống 5% mỗi cái
        #            để Agent không lãng phí capacity vào bài quá dễ
        #   Phase 4: Loại Stage 1, tập trung Stage 5-6, giữ rehearsal Stage 2-4
        #   Phase 5: Graduation — Stage 6 cao nhất (25%) vì là màn khó nhất;
        #            Stage 1-3 mỗi cái 10% đủ để neo bộ nhớ dài hạn
        self.phase_config: dict[int, tuple[list, list]] = {
            1: ([1, 2, 3],          [0.85, 0.10, 0.05]),  # Cold-Start: 85% Stage 1 trước
            2: ([1, 2, 3, 4],       [0.10, 0.10, 0.50, 0.30]),
            3: ([1, 2, 3, 4, 5],    [0.05, 0.05, 0.10, 0.45, 0.35]),
            4: ([2, 3, 4, 5, 6],    [0.10, 0.10, 0.10, 0.30, 0.40]),
            5: ([1, 2, 3, 4, 5, 6], [0.10, 0.10, 0.15, 0.20, 0.20, 0.25]),
        }

        # Trạng thái reward_scale hiện tại (được set khi reset())
        self.reward_scale: float = 1.0

    # ------------------------------------------------------------------
    # Rolling Window Win Rate
    # ------------------------------------------------------------------

    def get_win_rate(self, stage: int) -> float:
        """Tính Win Rate của một Stage dựa trên Rolling Window 200 episodes."""
        hist = self.stage_history.get(stage)
        if hist is None or len(hist) == 0:
            return 0.0
        return float(np.mean(hist))

    def get_win_rate_summary(self) -> dict[int, float]:
        """Trả về dict {stage: win_rate} cho tất cả Stage có dữ liệu."""
        return {
            s: self.get_win_rate(s)
            for s in range(1, 7)
            if len(self.stage_history[s]) > 0
        }

    # ------------------------------------------------------------------
    # Phase Promotion
    # ------------------------------------------------------------------

    def check_phase_promotion(self):
        """
        Kiểm tra điều kiện thăng Phase dựa trên Rolling Window Win Rate.
        Nếu thăng Phase, tự động kích hoạt Auto-Backup.
        """
        old_phase = self.train_phase

        if self.train_phase == 1:
            # Thăng Phase 2: Thành thạo Stage 1 và 2
            if self.get_win_rate(1) >= 0.90 and self.get_win_rate(2) >= 0.90:
                self.train_phase = 2

        elif self.train_phase == 2:
            # Thăng Phase 3: Ổn định Stage 3 và bước đầu Stage 4
            if self.get_win_rate(3) >= 0.80 and self.get_win_rate(4) >= 0.50:
                self.train_phase = 3

        elif self.train_phase == 3:
            # Thăng Phase 4: Ổn định Stage 4 và Stage 5 phải đạt 60%+
            # Nâng từ 50% → 60% để tránh Agent exploit pattern đơn giản
            # hoặc may mắn qua ngưỡng khi Stage 5 chỉ chiếm 35% data
            if self.get_win_rate(4) >= 0.75 and self.get_win_rate(5) >= 0.60:
                self.train_phase = 4

        elif self.train_phase == 4:
            # Thăng Phase 5 (Tốt nghiệp): Thành thạo Stage 5 và bước đầu Stage 6
            if self.get_win_rate(5) >= 0.80 and self.get_win_rate(6) >= 0.50:
                self.train_phase = 5

        # Kích hoạt Auto-Backup nếu phát hiện thăng Phase
        if self.train_phase != old_phase:
            print(
                f"\n🔥 [HỆ THỐNG] TỰ ĐỘNG THĂNG PHASE: Phase {old_phase} -> Phase {self.train_phase}!"
            )
            wr_summary = self.get_win_rate_summary()
            print(f"   Win Rates: { {s: f'{r:.1%}' for s, r in wr_summary.items()} }")
            # Cập nhật _curriculum_train_phase trên BulletHellEnv để get_phase_info() trả đúng
            if hasattr(self.env, "_curriculum_train_phase"):
                self.env._curriculum_train_phase = self.train_phase
            if self.model_saver_callback is not None:
                self.model_saver_callback(f"checkpoint_phase_{old_phase}_completed")

    # ------------------------------------------------------------------
    # update_post_episode — bắt buộc gọi cuối mỗi Episode
    # ------------------------------------------------------------------

    def update_post_episode(self, stage: int, is_win: bool):
        """
        Cập nhật Rolling Window sau khi một Episode kết thúc.
        Bắt buộc phải gọi trong vòng lặp huấn luyện chính hoặc từ step() khi done=True.

        :param stage: Stage của Episode vừa kết thúc (match_stage từ info dict)
        :param is_win: True nếu Agent thắng (Boss chết), False nếu thua
        """
        if stage in self.stage_history:
            self.stage_history[stage].append(1.0 if is_win else 0.0)
        self.check_phase_promotion()

    # ------------------------------------------------------------------
    # set_stage_parameters — giao tiếp với BulletHellEnv
    # ------------------------------------------------------------------

    def set_stage_parameters(self, stage: int, reward_scale: float):
        """
        Ép môi trường gốc thiết lập Stage và Reward Scale.
        Cập nhật _curriculum_train_phase trên BulletHellEnv để get_phase_info() trả Phase đúng.
        """
        self.current_stage = stage
        self.reward_scale = reward_scale

        if hasattr(self.env, "set_stage"):
            self.env.set_stage(stage)
            self.env.reward_scale = reward_scale
            # Truyền train_phase xuống BulletHellEnv để get_phase_info() trả đúng
            if hasattr(self.env, "_curriculum_train_phase"):
                self.env._curriculum_train_phase = self.train_phase
        else:
            self.env.current_stage = stage
            self.env.reward_scale = reward_scale

    def get_phase_info(self) -> dict:
        """
        Trả về train_phase thực sự của Wrapper.
        ActionMasker sẽ delegate method này xuống Wrapper qua __getattr__.
        """
        return {"train_phase": self.train_phase, "current_stage": self.current_stage}

    # ------------------------------------------------------------------
    # reset — chọn Stage theo Phase và áp Potential-based Reward Scaling
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        # 1. Lấy phân phối Stage cho Phase hiện tại
        stages, weights = self.phase_config.get(self.train_phase, self.phase_config[1])

        # 2. Chọn Stage ngẫu nhiên có trọng số
        chosen_stage = random.choices(stages, weights=weights)[0]
        self.current_stage = chosen_stage

        # 3. Potential-based Reward Scaling
        #    Stage 1 = ×1.0, Stage 6 = ×2.25 (tăng 0.25 mỗi Stage)
        stage_reward_scale = 1.0 + (chosen_stage - 1) * 0.25
        self.reward_scale = stage_reward_scale

        # 4. Ép môi trường gốc nhận Stage và Reward Scale
        self.set_stage_parameters(stage=chosen_stage, reward_scale=stage_reward_scale)

        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            is_win = info.get("is_win", False)
            self.update_post_episode(self.current_stage, is_win)
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Thuộc tính proxy — đảm bảo train_ppo.py vẫn tương thích
    # ------------------------------------------------------------------

    def set_stage(self, stage: int):
        """Proxy cho train_ppo.py gọi env_method('set_stage', stage)."""
        if hasattr(self.env, "set_stage"):
            self.env.set_stage(stage)
        self.current_stage = stage

    def set_rollback_buffer(self, active: bool):
        """Proxy cho train_ppo.py gọi env_method('set_rollback_buffer', active)."""
        if hasattr(self.env, "set_rollback_buffer"):
            self.env.set_rollback_buffer(active)

    def action_masks(self) -> np.ndarray:
        """Proxy cho ActionMasker."""
        return self.env.action_masks()

    @property
    def in_rollback_buffer(self) -> bool:
        return getattr(self.env, "in_rollback_buffer", False)

    @in_rollback_buffer.setter
    def in_rollback_buffer(self, value: bool):
        if hasattr(self.env, "in_rollback_buffer"):
            self.env.in_rollback_buffer = value

    def __repr__(self) -> str:
        wr = self.get_win_rate_summary()
        return (
            f"AutomatedCurriculumWrapper("
            f"phase={self.train_phase}, stage={self.current_stage}, "
            f"win_rates={wr})"
        )
