"""
BossBulletManager — quản lý đạn do Boss bắn ra.

Đạn có velocity 2D (vx, vy) và tùy chọn homing (đuổi theo player).
"""

import math

from config.settings import WIDTH, HEIGHT


class BossBullet:
    """Viên đạn của Boss — có velocity 2D, hỗ trợ homing."""

    RADIUS = 8

    def __init__(self, x: float, y: float, vx: float, vy: float, homing: bool = False):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.homing = homing
        self._speed = math.hypot(vx, vy)

    def update(self, player_cx: float = 0.0, player_cy: float = 0.0):
        if self.homing and self._speed > 0:
            # Xoay nhẹ về phía player (góc hội tụ tối đa 3°/frame)
            dx = player_cx - self.x
            dy = player_cy - self.y
            dist = math.hypot(dx, dy) or 1.0
            target_vx = dx / dist * self._speed
            target_vy = dy / dist * self._speed
            # Interpolation nhẹ để không quay quá gắt
            alpha = 0.04
            self.vx += (target_vx - self.vx) * alpha
            self.vy += (target_vy - self.vy) * alpha
            # Giữ speed cố định
            cur = math.hypot(self.vx, self.vy) or 1.0
            self.vx = self.vx / cur * self._speed
            self.vy = self.vy / cur * self._speed

        self.x += self.vx
        self.y += self.vy

    def is_outside_screen(self) -> bool:
        margin = 50
        return (
            self.x < -margin or self.x > WIDTH + margin
            or self.y < -margin or self.y > HEIGHT + margin
        )

    def draw(self, screen):
        import pygame
        pygame.draw.circle(
            screen,
            (255, 80, 80),
            (int(self.x), int(self.y)),
            self.RADIUS
        )


class BossBulletManager:

    def __init__(self):
        self.bullets: list[BossBullet] = []

    def spawn_from_dicts(self, bullet_dicts: list):
        """
        Nhận list[dict] từ BossPPO.update() và tạo BossBullet.
        Mỗi dict: {"x", "y", "vx", "vy", "homing"(optional)}.
        """
        for d in bullet_dicts:
            self.bullets.append(BossBullet(
                x=d["x"], y=d["y"],
                vx=d["vx"], vy=d["vy"],
                homing=d.get("homing", False)
            ))

    def update(self, player_cx: float = 0.0, player_cy: float = 0.0):
        for b in self.bullets:
            b.update(player_cx, player_cy)
        self.bullets = [b for b in self.bullets if not b.is_outside_screen()]

    def draw(self, screen):
        for b in self.bullets:
            b.draw(screen)
