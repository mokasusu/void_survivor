"""
BulletHellEnv — Gymnasium environment cho MaskablePPO.

Tuân thủ đặc tả void_survivor.md:
  - Observation: 45-dim vector chuẩn hóa [-1, 1]
  - Action: Discrete(9) — 0=idle, 1-4=move, 5-8=move+shoot
  - ActionMasking: khóa action 5-8 khi cooldown > 0
  - Stage Mixing: 70/20/10 tại mỗi reset()
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

        # Curriculum state — được set bởi train_ppo.py
        self.current_stage: int = 1

        # Internals
        self._encoder = PPOStateEncoder()
        self._reward_shaper = PPORewardShaper()
        self._steps: int = 0

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
    # Stage Mixing (70 / 20 / 10)
    # ------------------------------------------------------------------

    def _pick_stage(self) -> int:
        if self.current_stage == 1:
            return 1
        r = np.random.rand()
        if r < 0.70:
            return self.current_stage
        elif r < 0.90:
            return max(1, self.current_stage - 1)
        else:
            return int(np.random.randint(1, max(2, self.current_stage)))

    # ------------------------------------------------------------------
    # set_stage — gọi từ train_ppo qua vec_env.env_method
    # ------------------------------------------------------------------

    def set_stage(self, stage: int):
        self.current_stage = max(1, min(4, stage))

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

        self._active_stage = self._pick_stage()
        self.game.init_match(stage=self._active_stage)
        self._reward_shaper.reset(self.game)
        self._steps = 0

        obs = self._encoder.encode(self.game)
        info = {"stage": self._active_stage}
        return obs, info

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: int):
        action_enum, is_shooting = ACTION_MAP[int(action)]
        self.game.update_with_action(action_enum, is_shooting)
        self._steps += 1

        obs = self._encoder.encode(self.game)
        reward, breakdown = self._reward_shaper.compute(self.game)

        terminated = self.game.is_over()
        truncated = self._steps >= self.max_steps

        info = {
            "is_win": self.game.is_victory,
            "match_stage": self._active_stage,
            "reward_breakdown": breakdown,
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
