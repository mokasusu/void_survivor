"""
BossPPO — phiên bản Boss tích hợp Curriculum Stage.

Hỗ trợ:
  - Stage 1: đứng yên, HP 100, không bắn.
  - Stage 2: đứng yên, HP 150, bắn thẳng 2s/viên.
  - Stage 3: di chuyển ngang, HP 200, bắn thẳng 1s/viên, đạn nhanh.
  - Stage 4: di chuyển + đổi hướng thông minh, HP 300, spread+homing.
"""

import pygame
import random

from config.settings import WIDTH, HEIGHT, INFO_PANEL_HEIGHT
from core.assets import load_image
from entities.boss_stage import get_stage_config, spawn_boss_bullets


class BossPPO:

    DISPLAY_WIDTH = 120
    DISPLAY_HEIGHT = 120

    def __init__(self, stage: int = 1, stage_steps: int = 0):
        self.sprite = load_image("boss.png")

        self.display_width = self.DISPLAY_WIDTH
        self.display_height = self.DISPLAY_HEIGHT

        if self.sprite is not None:
            w, h = self.sprite.get_size()
            scale = self.display_width / max(w, 1)
            self.display_width = int(w * scale)
            self.display_height = int(h * scale)
            self.sprite = pygame.transform.smoothscale(
                self.sprite,
                (self.display_width, self.display_height)
            )

        self.x = 70
        self.y = INFO_PANEL_HEIGHT + 40

        # Movement state
        self._move_dir = 1          # +1 xuống, -1 lên
        self._move_timer = 0
        self._move_change_interval = 90   # frames trước khi đổi hướng (Stage 3+)
        self._smart_timer = 0             # cho Stage 4

        # Shoot state
        self._shoot_timer = 0

        self.init_stage(stage, stage_steps)

    # ------------------------------------------------------------------
    # Stage initialization
    # ------------------------------------------------------------------

    def init_stage(self, stage: int, stage_steps: int = 0):
        cfg = get_stage_config(stage)
        self._stage = stage
        self._cfg = cfg

        self.max_health = cfg["hp"]
        self.health = cfg["hp"]

        self._moves = cfg["moves"]
        self._move_speed = cfg["move_speed"]
        self._shoot_enabled = cfg["shoot"]
        self._shoot_interval = cfg["shoot_interval_frames"]
        self._bullet_speed = cfg["bullet_speed"]
        self._pattern = cfg["pattern"]

        # --- DDA (Dynamic Difficulty Adjustment) dựa trên stage_steps ---
        # 12,500 steps/env tương ứng với 100,000 global steps khi chạy 8 envs
        transition_steps = 12500
        p = min(1.0, max(0.0, stage_steps / transition_steps))

        if stage == 2:
            # Soft Start: Bắn thưa ở đầu Stage 2 (300 frames ~ 5 giây) rồi tăng dần về mặc định (120 frames)
            self._shoot_interval = int(300 - (300 - cfg["shoot_interval_frames"]) * p)
        elif stage == 3:
            # Boss di chuyển chậm ở đầu (0.5) rồi nhanh dần về mặc định (2.0)
            self._move_speed = 0.5 + (cfg["move_speed"] - 0.5) * p
            # Thu hẹp biên độ di chuyển ở đầu (30% biên độ dọc) rồi mở rộng dần về mặc định (100%)
            self._move_range_scale = 0.3 + 0.7 * p
        else:
            self._move_range_scale = 1.0

        # Reset timers
        self._shoot_timer = 0
        self._move_timer = 0
        self._move_dir = random.choice([-1, 1])

        # Đặt Boss ở vị trí ngẫu nhiên cột bên trái
        self.x = 70
        min_y = INFO_PANEL_HEIGHT
        max_y = HEIGHT - self.display_height
        c_y = (min_y + max_y) / 2
        r_y = (max_y - min_y) / 2
        if stage == 3:
            min_y = int(c_y - r_y * self._move_range_scale)
            max_y = int(c_y + r_y * self._move_range_scale)
        self.y = random.randint(min_y, max_y)

    # ------------------------------------------------------------------
    # Update — trả về list các dict đạn cần spawn
    # ------------------------------------------------------------------

    def update(self, player_cx: float, player_cy: float) -> list:
        """
        Cập nhật vị trí Boss và tính đạn cần sinh.
        Trả về list[dict] — mỗi dict = 1 viên đạn mới.
        """
        self._update_movement(player_cx, player_cy)
        bullets = self._update_shoot(player_cx, player_cy)
        return bullets

    def _update_movement(self, player_cx: float, player_cy: float):
        if not self._moves:
            return

        min_y = INFO_PANEL_HEIGHT
        max_y = HEIGHT - self.display_height

        if self._stage == 4:
            # Stage 4: di chuyển thông minh — đuổi theo Y của player
            self._smart_timer += 1
            boss_cy = self.y + self.display_height / 2
            if boss_cy < player_cy:
                self._move_dir = 1
            else:
                self._move_dir = -1
        else:
            # Stage 3: qua lại tuần hoàn
            self._move_timer += 1
            if self._move_timer >= self._move_change_interval:
                self._move_dir *= -1
                self._move_timer = 0

            # Áp dụng biên độ di chuyển thu hẹp cho Stage 3
            if self._stage == 3 and hasattr(self, "_move_range_scale"):
                c_y = (min_y + max_y) / 2
                r_y = (max_y - min_y) / 2
                min_y = c_y - r_y * self._move_range_scale
                max_y = c_y + r_y * self._move_range_scale

        self.y += self._move_dir * self._move_speed
        self.y = max(min_y, min(max_y, self.y))

    def _update_shoot(self, player_cx: float, player_cy: float) -> list:
        if not self._shoot_enabled or self._shoot_interval <= 0:
            return []

        self._shoot_timer += 1
        if self._shoot_timer < self._shoot_interval:
            return []

        self._shoot_timer = 0
        boss_cx = self.x + self.display_width / 2
        boss_cy = self.y + self.display_height / 2
        return spawn_boss_bullets(
            self._pattern, boss_cx, boss_cy,
            player_cx, player_cy, self._bullet_speed
        )

    # ------------------------------------------------------------------
    # Damage / helpers
    # ------------------------------------------------------------------

    def take_damage(self, amount: int = 1):
        self.health = max(0, self.health - amount)

    def get_rect(self) -> pygame.Rect:
        return pygame.Rect(self.x, self.y, self.display_width, self.display_height)

    def get_center(self):
        return (self.x + self.display_width / 2, self.y + self.display_height / 2)

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------

    def draw(self, screen):
        if self.sprite is not None:
            screen.blit(self.sprite, (self.x, self.y))
            return
        pygame.draw.rect(
            screen,
            (220, 60, 60),
            (self.x, self.y, self.display_width, self.display_height)
        )
