import random

from entities.bullet import Bullet

from config.settings import HEIGHT
from config.settings import INFO_PANEL_HEIGHT


class BulletManager:

    def __init__(self):

        self.bullets = []

    def spawn_bullet(self):

        y = random.randint(
            INFO_PANEL_HEIGHT,
            HEIGHT
        )

        speed = random.randint(5, 12)

        bullet = Bullet(
            0,
            y,
            speed
        )

        self.bullets.append(bullet)

    def update(self):

        for bullet in self.bullets:

            bullet.update()

        self.bullets = [
            bullet
            for bullet in self.bullets
            if not bullet.is_outside_screen()
        ]

    def draw(self, screen):

        for bullet in self.bullets:

            bullet.draw(screen)
