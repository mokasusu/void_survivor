import pygame
import math


class BlackHole:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.pull_strength = 0.18
        self.pull_radius = 380
        self.inner_radius = 28
        self.timer = 0
        self.spawn_time = pygame.time.get_ticks()

    def update(self):
        self.timer += 0.045

    def get_pull(self, player_x, player_y):
        dx = self.x - player_x
        dy = self.y - player_y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1:
            dist = 1
        if dist > self.pull_radius:
            return 0.0, 0.0
        # Lực hút tăng khi lại gần
        strength = self.pull_strength * (1.0 - dist / self.pull_radius) * 3.5
        return dx / dist * strength, dy / dist * strength

    def draw(self, screen):
        # Vòng sáng ngoài (glow)
        for i in range(6):
            ring_r = 55 + i * 18
            alpha = max(0, 90 - i * 14)
            surf = pygame.Surface((ring_r * 2, ring_r * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (90, 0, 160, alpha), (ring_r, ring_r), ring_r)
            screen.blit(surf, (int(self.x) - ring_r, int(self.y) - ring_r))

        # Tia xoáy
        for i in range(14):
            angle = self.timer + i * (math.pi * 2 / 14)
            r_vary = 42 + 14 * math.sin(self.timer * 2.2 + i)
            px = int(self.x + math.cos(angle) * r_vary)
            py = int(self.y + math.sin(angle) * r_vary)
            size = max(2, 5 - i // 4)
            pulse_color = (
                int(130 + 80 * math.sin(self.timer + i)),
                0,
                int(200 + 55 * math.cos(self.timer + i)),
            )
            pygame.draw.circle(screen, pulse_color, (px, py), size)

        # Lõi đen
        pygame.draw.circle(screen, (0, 0, 0), (int(self.x), int(self.y)), self.inner_radius)
        pygame.draw.circle(screen, (160, 0, 255), (int(self.x), int(self.y)), self.inner_radius, 3)

        # Nhãn cảnh báo
        elapsed = pygame.time.get_ticks() - self.spawn_time
        if elapsed < 2000:
            blink = (elapsed // 250) % 2
            if blink == 0:
                font = pygame.font.SysFont(["Orbitron", "Audiowide", "Consolas"], 16)
                label = font.render("VOID CORE", True, (200, 100, 255))
                screen.blit(label, (int(self.x) - label.get_width() // 2, int(self.y) - self.inner_radius - 24))
