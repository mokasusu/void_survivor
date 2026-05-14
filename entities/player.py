import pygame
import math

from controllers.action import Action

from config.settings import HEIGHT
from config.settings import WIDTH
from config.settings import INFO_PANEL_HEIGHT
from config.settings import PLAYER_COLOR
from core.assets import load_image


class Player:

    WIDTH = 30
    HEIGHT = 30
    DISPLAY_WIDTH = 64
    DISPLAY_HEIGHT = 51
    INVULNERABLE_DURATION_MS = 3000
    BLINK_INTERVAL_MS = 120

    def __init__(self, x, y):

        self.x = x
        self.y = y

        self.speed = 6

        self.max_health = 3
        self.health = 3
        self.weapon_level = 1
        self.invulnerable_until = 0
        self.sprite = load_image(
            "player.png",
            (self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT)
        )

    def upgrade_weapon(self):
        if self.weapon_level < 3:
            self.weapon_level += 1

    def update(self, action):

        if action == Action.UP:

            self.y -= self.speed

        elif action == Action.DOWN:

            self.y += self.speed

        elif action == Action.LEFT:

            self.x -= self.speed

        elif action == Action.RIGHT:

            self.x += self.speed

        self.clamp()

    def clamp(self):

        if self.y < INFO_PANEL_HEIGHT:

            self.y = INFO_PANEL_HEIGHT

        if self.y > HEIGHT - self.HEIGHT:

            self.y = HEIGHT - self.HEIGHT

        if self.x < 0:

            self.x = 0

        if self.x > WIDTH - self.WIDTH:

            self.x = WIDTH - self.WIDTH

    def is_invulnerable(self, current_time):

        return current_time < self.invulnerable_until

    def take_damage(self, current_time):

        if self.is_invulnerable(current_time):
            return False

        self.health -= 1
        self.invulnerable_until = (
            current_time + self.INVULNERABLE_DURATION_MS
        )
        return True

    def is_dead(self):

        return self.health <= 0

    def get_shoot_origin(self):

        return (
            self.x,
            self.y + (self.HEIGHT // 2)
        )

    def draw(self, screen):
        now = pygame.time.get_ticks()
        if self.is_invulnerable(now):
            blink_phase = (
                now // self.BLINK_INTERVAL_MS
            ) % 2
            if blink_phase == 0:
                return

        cx = self.x + self.WIDTH // 2
        cy = self.y + self.HEIGHT // 2

        if self.weapon_level == 2:
            pulse = 1.0 + 0.15 * math.sin(now * 0.006)
            aura_r = int(38 * pulse)
            aura_surf = pygame.Surface((aura_r * 2 + 4, aura_r * 2 + 4), pygame.SRCALPHA)
            pygame.draw.ellipse(aura_surf, (255, 220, 50, 80), (0, 0, aura_r * 2 + 4, aura_r * 2 + 4))
            pygame.draw.ellipse(aura_surf, (255, 220, 50, 160), (4, 4, aura_r * 2 - 4, aura_r * 2 - 4), 3)
            screen.blit(aura_surf, (cx - aura_r - 2, cy - aura_r - 2))

        elif self.weapon_level >= 3:
            pulse = 1.0 + 0.2 * math.sin(now * 0.007)
            aura_r = int(46 * pulse)
            for i in range(3):
                alpha = 120 - i * 35
                r = aura_r - i * 7
                if r <= 0:
                    continue
                aura_surf = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
                pygame.draw.ellipse(aura_surf, (50, 230, 255, alpha), (0, 0, r * 2 + 4, r * 2 + 4))
                screen.blit(aura_surf, (cx - r - 2, cy - r - 2))

        if self.sprite is not None:
            draw_x = self.x - (
                (self.DISPLAY_WIDTH - self.WIDTH) // 2
            )
            draw_y = self.y - (
                (self.DISPLAY_HEIGHT - self.HEIGHT) // 2
            )
            screen.blit(
                self.sprite,
                (draw_x, draw_y)
            )
            return

        pygame.draw.rect(
            screen,
            PLAYER_COLOR,
            (
                self.x,
                self.y,
                self.WIDTH,
                self.HEIGHT
            )
        )
