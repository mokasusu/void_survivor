import pygame
import random
import math

from config.settings import WIDTH, HEIGHT, INFO_PANEL_HEIGHT
from core.assets import load_image
from entities.laser import Laser
from entities.minion import Minion
from entities.bullet import Bullet


class Boss:

    def __init__(self):
        self.sprite = load_image("boss.png")
        self.max_health = 100
        self.health = 100
        self.display_width = 120
        self.display_height = 120

        if self.sprite is not None:
            width, height = self.sprite.get_size()
            scale = self.display_width / max(width, 1)
            self.display_width = int(width * scale)
            self.display_height = int(height * scale)
            self.sprite = pygame.transform.smoothscale(
                self.sprite,
                (self.display_width, self.display_height)
            )

        self.x = 70
        self.y = INFO_PANEL_HEIGHT + 40
        self.speed_y = 3

        self.lasers = []
        self.minions = []
        self.new_bullets = []

        self.attack_timer = 180
        self.minions_spawned = False

        # Void / Enrage flags
        self.void_active = False        # Black hole phase (30-35% HP)
        self.enraged = False            # Enrage phase (15% HP)
        self.enrage_timer = 0

    def _spawn_minions(self):
        """Gọi ra 2-3 đệ tử xoay quanh Boss."""
        count = random.randint(2, 3)
        cx = self.x + self.display_width // 2
        cy = self.y + self.display_height // 2
        for i in range(count):
            angle = i * (2 * math.pi / count)
            self.minions.append(Minion(cx, cy, angle))

    def update(self, player_x=None, player_y=None):
        # --- Di chuyển ---
        speed = self.speed_y * (1.5 if self.enraged else 1.0)
        self.y += speed
        if self.y < INFO_PANEL_HEIGHT:
            self.y = INFO_PANEL_HEIGHT
            self.speed_y = abs(self.speed_y)
        elif self.y > HEIGHT - self.display_height:
            self.y = HEIGHT - self.display_height
            self.speed_y = -abs(self.speed_y)

        # --- Triệu hồi đệ tử khi Boss còn 85 HP ---
        if self.health <= 85 and not self.minions_spawned:
            self.minions_spawned = True
            self._spawn_minions()

        # --- Cập nhật đệ tử + bắn đạn nhắm player ---
        cx = self.x + self.display_width // 2
        cy = self.y + self.display_height // 2
        for minion in self.minions:
            minion.update(cx, cy)
            if player_x is not None and player_y is not None:
                shot = minion.get_aimed_shot(player_x, player_y)
                if shot is not None:
                    bx, by, sx, sy = shot
                    self.new_bullets.append(Bullet(bx, by, sx, sy))

        self.minions = [m for m in self.minions if m.health > 0]

        # --- Bộ đếm tấn công ---
        cooldown = int(self.attack_timer * (0.65 if self.enraged else 1.0))
        cooldown = max(60, cooldown)
        # (attack_timer đếm ngược từ giá trị random, ta dùng self._atk_cd nội bộ)
        if not hasattr(self, "_atk_cd"):
            self._atk_cd = cooldown

        self._atk_cd -= 1
        if self._atk_cd <= 0:
            self._atk_cd = random.randint(
                int(100 * (0.6 if self.enraged else 1.0)),
                int(160 * (0.6 if self.enraged else 1.0)),
            )
            if random.random() < 0.5:
                self.lasers.append(Laser(self.x + self.display_width, cy))
            else:
                spd = 8 if self.enraged else 7
                for dy_off in (-2, 0, 2):
                    self.new_bullets.append(Bullet(self.x + self.display_width, cy, spd, dy_off))

        # Cập nhật laser
        for laser in self.lasers:
            laser.update(self.x + self.display_width, cy)
        self.lasers = [l for l in self.lasers if l.state != "done"]

        # --- Enrage timer (hiệu ứng nhấp nháy) ---
        if self.enraged:
            self.enrage_timer += 1

    def enter_void_phase(self):
        self.void_active = True

    def enter_enrage(self):
        if not self.enraged:
            self.enraged = True
            self.health = min(self.max_health, self.health + 10)  # Hồi 10 HP

    def draw(self, screen):
        for laser in self.lasers:
            laser.draw(screen)

        for minion in self.minions:
            minion.draw(screen)

        if self.sprite is not None:
            draw_sprite = self.sprite
            # Khi enraged: tô đỏ bằng cách blend
            if self.enraged:
                blink = (self.enrage_timer // 8) % 2
                if blink == 0:
                    red_surf = self.sprite.copy()
                    red_overlay = pygame.Surface(red_surf.get_size(), pygame.SRCALPHA)
                    red_overlay.fill((255, 0, 0, 100))
                    red_surf.blit(red_overlay, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
                    draw_sprite = red_surf
            screen.blit(draw_sprite, (self.x, self.y))
        else:
            color = (255, 60, 60) if not self.enraged else (255, 0, 0)
            pygame.draw.rect(screen, color, (self.x, self.y, self.display_width, self.display_height))

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.display_width, self.display_height)

    def take_damage(self, amount=1):
        self.health = max(0, self.health - amount)
