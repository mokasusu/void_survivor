import pygame
import random

from config.settings import WIDTH
from config.settings import HEIGHT
from config.settings import INFO_PANEL_HEIGHT
from core.assets import load_image


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

    def draw(self, screen):

        if self.sprite is not None:
            screen.blit(
                self.sprite,
                (self.x, self.y)
            )
            return

        pygame.draw.rect(
            screen,
            (220, 60, 60),
            (
                self.x,
                self.y,
                self.display_width,
                self.display_height
            )
        )

    def get_rect(self):

        return pygame.Rect(
            self.x,
            self.y,
            self.display_width,
            self.display_height
        )

    def take_damage(self, amount=1):

        self.health = max(0, self.health - amount)

    def randomize_vertical_position(self):

        min_y = INFO_PANEL_HEIGHT
        max_y = HEIGHT - self.display_height
        self.y = random.randint(min_y, max_y)
