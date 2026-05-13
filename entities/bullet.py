import pygame

from config.settings import BULLET_COLOR
from core.assets import load_image


class Bullet:

    RADIUS = 10
    SIZE = 64

    def __init__(self, x, y, speed):

        self.x = x
        self.y = y

        self.speed = speed
        self.sprite = load_image(
            "bullet.png",
            (self.SIZE, self.SIZE)
        )

    def update(self):

        self.x += self.speed

    def is_outside_screen(self):

        return self.x > 900

    def draw(self, screen):
        if self.sprite is not None:
            screen.blit(
                self.sprite,
                (
                    int(self.x - self.SIZE // 2),
                    int(self.y - self.SIZE // 2)
                )
            )
            return

        pygame.draw.circle(
            screen,
            BULLET_COLOR,
            (
                int(self.x),
                int(self.y)
            ),
            self.RADIUS
        )
