"""
BulletHellEnv — Gymnasium environment cho MaskablePPO.

Tuân thủ đặc tả void_survivor.md:
  - Observation: 45-dim vector chuẩn hóa [-1, 1]
  - Action: Discrete(9) — 0=idle, 1-4=move, 5-8=move+shoot
  - ActionMasking: khóa action 5-8 khi cooldown > 0
  - Stage Mixing: "Sliding Window" — 60/25/15 tại mỗi reset()
  - Adaptive Stagnation: ngưỡng và penalty co giãn theo stage
"""

import os
import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

from core.game_ppo import GamePPO
from controllers.action import Action
from rl.ppo_state_encoder import PPOStateEncoder
from rl.ppo_rewards import PPORewardShaper
from config.settings import WIDTH, HEIGHT


# Action index → (Action enum, is_shooting)
ACTION_MAP = {
    0: (Action.IDLE,  False),
    1: (Action.UP,    False),
    2: (Action.DOWN,  False),
    3: (Action.LEFT,  False),
    4: (Action.RIGHT, False),
    5: (Action.UP,    True),
    6: (Action.DOWN,  True),
    7: (Action.LEFT,  True),
    8: (Action.RIGHT, True),
}

# ---------------------------------------------------------------------------
# Adaptive stagnation config theo stage
# Ngưỡng frame không gây damage trước khi truncate trận (không phạt điểm âm nặng)
# Stage cao hơn → cho agent nhiều thời gian hơn để né đạn
# ---------------------------------------------------------------------------
_STAGNATION_TRUNCATE_FRAMES = {
    1:  600,   # ~10 giây
    2:  900,   # ~15 giây
    3: 1200,   # ~20 giây
    4: 2000,   # ~33 giây
    5: 2500,   # ~41 giây
    6: 3000,   # ~50 giây
}
# Penalty khi bị truncate do trì trệ — stage cao phạt nhẹ hơn
# (Không phạt quá nặng để tránh suicide loop)
_STAGNATION_TRUNCATE_PENALTY = {
    1: -5.0,
    2: -3.0,
    3: -2.0,
    4: -1.0,
    5: -0.5,
    6: -0.5,
}


class BulletHellEnv(gym.Env):
    """
    Custom Gymnasium env bọc GamePPO.
    Dùng với sb3_contrib.common.wrappers.ActionMasker.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        render_mode: str | None = None,
        max_steps: int = 3000,
        render_fps: int = 60,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.render_fps = render_fps

        # Spaces
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(45,), dtype=np.float32
        )

        # Curriculum state — được set bởi train_ppo.py hoặc AutomatedCurriculumWrapper
        self.current_stage: int = 1
        self.steps_in_stage: int = 0
        self.in_rollback_buffer: bool = False

        # Potential-based Reward Scaling — được set bởi AutomatedCurriculumWrapper
        # Stage 1 = 1.0×, Stage 6 = 2.25× (tăng 0.25 mỗi Stage)
        self.reward_scale: float = 1.0

        # Stage override — set bởi AutomatedCurriculumWrapper.reset() trước khi gọi env.reset()
        # Đảm bảo _pick_stage() không override lại Stage mà Wrapper đã chọn
        self._active_stage_override: int | None = None

        # Train phase hiện tại — được Wrapper cập nhật để get_phase_info() trả đúng giá trị
        self._curriculum_train_phase: int = 1

        # Internals
        self._encoder = PPOStateEncoder()
        self._reward_shaper = PPORewardShaper()
        self._steps: int = 0
        self.frames_since_last_damage = 0

        # Pygame setup
        if render_mode != "human":
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        pygame.init()
        self._screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self._clock = pygame.time.Clock() if render_mode == "human" else None

        sim_step_ms = int(1000 / max(1, render_fps))
        self.game = GamePPO(
            use_sim_time=(render_mode != "human"),
            sim_step_ms=sim_step_ms,
        )
        self._active_stage: int = 1

    # ------------------------------------------------------------------
    # Stage Mixing — "Sliding Window" / Rollback Buffer
    # ------------------------------------------------------------------

    def _pick_stage(self) -> int:
        """
        Nếu ở Vùng đệm Rollback:
          50% → Stage hiện tại (danh nghĩa)
          50% → Stage tiếp theo (bản nâng cao đang học dở)
        Nếu bình thường, dùng Cửa sổ trượt (Sliding Window):
          60% → Stage hiện tại (bài mới)
          25% → Stage liền trước (ôn bài gần nhất)
          15% → Bất kỳ Stage nào trong quá khứ (neo bộ nhớ)

        Khi current_stage == 1, luôn trả về 1.
        Khi current_stage == 2, chia 70/30 (không có quá khứ xa).
        """
        s = self.current_stage
        if self.in_rollback_buffer:
            return s if np.random.rand() < 0.50 else min(6, s + 1)

        if s == 1:
            return 1
        if s == 2:
            return s if np.random.rand() < 0.70 else 1

        r = np.random.rand()
        if r < 0.60:
            return s                                          # Stage mới nhất
        elif r < 0.85:
            return s - 1                                      # Stage vừa qua
        else:
            return int(np.random.randint(1, s - 1))          # Quá khứ xa ngẫu nhiên

    # ------------------------------------------------------------------
    # set_stage / set_rollback_buffer — gọi từ train_ppo
    # ------------------------------------------------------------------

    def set_stage(self, stage: int):
        stage = max(1, min(6, stage))
        if self.current_stage != stage:
            self.current_stage = stage
            self.steps_in_stage = 0

    def set_rollback_buffer(self, active: bool):
        self.in_rollback_buffer = active

    def set_stage_parameters(self, stage: int, reward_scale: float):
        """
        Được gọi bởi AutomatedCurriculumWrapper để đồng bộ Stage và Reward Scale.
        Stage được lock vào _active_stage_override để reset() không gọi _pick_stage() override lại.
        """
        self.set_stage(stage)
        self.reward_scale = max(1.0, reward_scale)
        self._active_stage_override = stage  # Lock stage — reset() sẽ dùng cái này

    def get_phase_info(self) -> dict:
        """
        Trả về train_phase thực sự (AutomatedCurriculumWrapper sẽ cập nhật _curriculum_train_phase).
        VecEnv.env_method('get_phase_info') sẽ thu thập được thông tin Phase chính xác.
        """
        return {"train_phase": self._curriculum_train_phase, "current_stage": self.current_stage}

    # ------------------------------------------------------------------
    # ActionMasking
    # ------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        mask = np.ones(9, dtype=bool)
        if self.game.shoot_cooldown > 0:
            mask[5:9] = False   # Khóa UP/DOWN/LEFT/RIGHT + Shoot
        return mask

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Sử dụng override nếu AutomatedCurriculumWrapper đã chọn Stage trước,
        # ngược lại mới gọi _pick_stage() (để tương thích khi không có Wrapper)
        if self._active_stage_override is not None:
            self._active_stage = self._active_stage_override
            self._active_stage_override = None  # Clear sau khi dùng
        else:
            self._active_stage = self._pick_stage()

        # Điểm 1: stage_steps của trận = 0 khi bắt đầu mỗi trận mới.
        # Không lấy tổng thời gian tích lũy của toàn quá trình train (steps_in_stage)
        # áp vào 1 trận đơn lẻ — tránh "ép độ khó tối đa" vào bài ôn tập.
        self.game.init_match(stage=self._active_stage, stage_steps=0)
        self._reward_shaper.reset(self.game)
        self._steps = 0
        self.frames_since_last_damage = 0

        obs = self._encoder.encode(self.game)
        info = {"stage": self._active_stage}
        return obs, info

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: int):
        action_enum, is_shooting = ACTION_MAP[int(action)]

        # Lưu boss hp trước khi update
        prev_boss_hp = self.game.boss.health if self.game.boss else 0

        self.game.update_with_action(action_enum, is_shooting)
        self._steps += 1
        self.steps_in_stage += 1

        # Check xem boss có bị mất máu không
        cur_boss_hp = self.game.boss.health if self.game.boss else 0
        is_hit_boss = (prev_boss_hp - cur_boss_hp) > 0

        if is_hit_boss:
            self.frames_since_last_damage = 0
        else:
            self.frames_since_last_damage += 1

        obs = self._encoder.encode(self.game)
        reward, breakdown = self._reward_shaper.compute(
            self.game,
            stage=self._active_stage,
            stage_steps=self._steps,   # stage_steps = số bước trong trận này, không phải toàn bộ stage
            reward_scale=self.reward_scale,  # Potential-based Reward Scaling từ AutomatedCurriculumWrapper
        )

        terminated = self.game.is_over()
        truncated = self._steps >= self.max_steps

        # Điểm 2: Adaptive stagnation — ngưỡng và penalty co giãn theo stage
        stagnation_limit = _STAGNATION_TRUNCATE_FRAMES.get(self._active_stage, 800)
        stagnation_penalty_val = 0.0
        if not truncated and self.frames_since_last_damage > stagnation_limit:
            truncated = True
            stagnation_penalty_val = _STAGNATION_TRUNCATE_PENALTY.get(self._active_stage, -2.0)
            reward += stagnation_penalty_val

        breakdown["stagnation_truncate_penalty"] = stagnation_penalty_val

        info = {
            "is_win": self.game.is_victory,
            "match_stage": self._active_stage,
            "reward_breakdown": breakdown,
            "frames_since_last_damage": self.frames_since_last_damage,
        }

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _render_frame(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.game.running = False
        self.game.draw(self._screen)
        pygame.display.flip()
        if self._clock:
            self._clock.tick(self.render_fps)

    def render(self):
        if self.render_mode == "human":
            self._render_frame()

    def close(self):
        pygame.quit()
