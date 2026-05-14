import pygame
import math
import random


class Minion:
    def __init__(self, center_x, center_y, angle_offset):
        self.center_x = center_x
        self.center_y = center_y
        self.angle = angle_offset
        self.radius = 75
        self.orbit_speed = 0.045
        self.x = self.center_x + math.cos(self.angle) * self.radius
        self.y = self.center_y + math.sin(self.angle) * self.radius
        self.health = 10
        self.width = 22
        self.height = 22
        self.shoot_timer = random.randint(80, 140)
        self.pulse_timer = 0

    def update(self, center_x, center_y):
        self.center_x = center_x
        self.center_y = center_y
        self.angle += self.orbit_speed
        self.x = self.center_x + math.cos(self.angle) * self.radius
        self.y = self.center_y + math.sin(self.angle) * self.radius
        self.shoot_timer -= 1
        self.pulse_timer += 0.1

    def get_rect(self):
        return pygame.Rect(
            self.x - self.width // 2,
            self.y - self.height // 2,
            self.width,
            self.height,
        )

    def take_damage(self, amount=1):
        self.health -= amount

    def get_aimed_shot(self, target_x, target_y):
        """Trả về (bx, by, speed_x, speed_y) nếu đến lúc bắn, else None."""
        if self.shoot_timer > 0:
            return None
        self.shoot_timer = random.randint(90, 150)
        dx = target_x - self.x
        dy = target_y - self.y
        dist = math.sqrt(dx * dx + dy * dy) or 1
        spd = 5.5
        return (self.x, self.y, dx / dist * spd, dy / dist * spd)

    def draw(self, screen):
        # Glow pulse
        pulse = 1.0 + 0.2 * math.sin(self.pulse_timer)
        glow_r = int(18 * pulse)
        glow_surf = pygame.Surface((glow_r * 2, glow_r * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (220, 80, 255, 80), (glow_r, glow_r), glow_r)
        screen.blit(glow_surf, (int(self.x) - glow_r, int(self.y) - glow_r))

        # Thân (hình thoi nhỏ)
        cx, cy = int(self.x), int(self.y)
        hw, hh = self.width // 2, self.height // 2
        points = [(cx, cy - hh), (cx + hw, cy), (cx, cy + hh), (cx - hw, cy)]
        pygame.draw.polygon(screen, (200, 80, 255), points)
        pygame.draw.polygon(screen, (255, 180, 255), points, 2)

        # Lõi sáng
        pygame.draw.circle(screen, (255, 220, 255), (cx, cy), 4)
