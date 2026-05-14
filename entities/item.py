import pygame
import math
from config.settings import WIDTH

class Item:
    def __init__(self, x, y, item_type):
        self.x = x
        self.y = y
        self.type = item_type
        self.speed = 2
        self.base_width = 22
        self.base_height = 22
        self.float_timer = 0
        self.pulse_timer = 0
        self.angle = 0

    def update(self):
        self.x += self.speed
        self.float_timer += 0.08
        self.y += math.sin(self.float_timer) * 1.2
        self.pulse_timer += 0.12
        self.angle = (self.angle + 2) % 360

    def is_outside_screen(self):
        return self.x > WIDTH + 50

    def get_rect(self):
        return pygame.Rect(
            self.x - self.base_width // 2,
            self.y - self.base_height // 2,
            self.base_width,
            self.base_height
        )

    def draw(self, screen):
        pulse_scale = 1.0 + 0.18 * math.sin(self.pulse_timer)
        w = int(self.base_width * pulse_scale)
        h = int(self.base_height * pulse_scale)

        if self.type == "heart":
            self._draw_heart(screen, w, h)
        elif self.type == "weapon":
            self._draw_weapon_gem(screen, w, h)

    def _draw_heart(self, screen, w, h):
        surf = pygame.Surface((w + 10, h + 10), pygame.SRCALPHA)
        cx = (w + 10) // 2
        cy = (h + 10) // 2

        # Glow
        glow_surf = pygame.Surface((w + 20, h + 20), pygame.SRCALPHA)
        pygame.draw.ellipse(glow_surf, (255, 80, 80, 60), (0, 0, w + 20, h + 20))
        screen.blit(glow_surf, (int(self.x) - (w + 20) // 2, int(self.y) - (h + 20) // 2))

        # Main heart shape using circles + triangle
        r = w // 4
        pygame.draw.circle(surf, (255, 50, 80), (cx - r, cy - r // 2), r)
        pygame.draw.circle(surf, (255, 50, 80), (cx + r, cy - r // 2), r)
        points = [
            (cx - w // 2 + 2, cy - r // 2),
            (cx + w // 2 - 2, cy - r // 2),
            (cx, cy + h // 2 - 2)
        ]
        pygame.draw.polygon(surf, (255, 50, 80), points)

        # Shine
        pygame.draw.circle(surf, (255, 200, 210, 180), (cx - r // 2, cy - r), r // 3)

        screen.blit(surf, (int(self.x) - (w + 10) // 2, int(self.y) - (h + 10) // 2))

    def _draw_weapon_gem(self, screen, w, h):
        # Rotating diamond gem
        surf = pygame.Surface((w + 20, h + 20), pygame.SRCALPHA)
        cx = (w + 20) // 2
        cy = (h + 20) // 2

        # Glow
        glow_surf = pygame.Surface((w + 30, h + 30), pygame.SRCALPHA)
        pygame.draw.ellipse(glow_surf, (50, 255, 120, 60), (0, 0, w + 30, h + 30))
        screen.blit(glow_surf, (int(self.x) - (w + 30) // 2, int(self.y) - (h + 30) // 2))

        # Diamond shape
        angle_rad = math.radians(self.angle)
        half = w // 2
        points = [
            (cx + half * math.cos(angle_rad), cy + half * math.sin(angle_rad)),
            (cx + half * math.cos(angle_rad + math.pi / 2), cy + half * math.sin(angle_rad + math.pi / 2)),
            (cx + half * math.cos(angle_rad + math.pi), cy + half * math.sin(angle_rad + math.pi)),
            (cx + half * math.cos(angle_rad + 3 * math.pi / 2), cy + half * math.sin(angle_rad + 3 * math.pi / 2)),
        ]
        pygame.draw.polygon(surf, (50, 230, 100), points)

        # Shine
        shine_x = cx + (half // 3) * math.cos(angle_rad - 0.5)
        shine_y = cy + (half // 3) * math.sin(angle_rad - 0.5)
        pygame.draw.circle(surf, (200, 255, 220, 200), (int(shine_x), int(shine_y)), max(2, w // 8))

        screen.blit(surf, (int(self.x) - (w + 20) // 2, int(self.y) - (h + 20) // 2))
