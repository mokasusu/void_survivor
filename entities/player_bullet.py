import pygame

from core.assets import load_image


class PlayerBullet:

    RADIUS = 8
    SIZE = 44

    def __init__(self, x, y, speed):

        self.x = x
        self.y = y
        self.speed = speed
        self.sprite = load_image("player_bullet.png")
        if self.sprite is not None:
            width, height = self.sprite.get_size()
            scale = self.SIZE / max(width, height)
            target_size = (
                max(1, int(width * scale)),
                max(1, int(height * scale))
            )
            self.sprite = pygame.transform.smoothscale(
                self.sprite,
                target_size
            )

    def update(self):

        self.x += self.speed

    def is_outside_screen(self):

        return self.x < -50

    def draw(self, screen):

        if self.sprite is not None:
            sprite_w, sprite_h = self.sprite.get_size()
            screen.blit(
                self.sprite,
                (
                    int(self.x - sprite_w // 2),
                    int(self.y - sprite_h // 2)
                )
            )
            return

        pygame.draw.circle(
            screen,
            (80, 200, 255),
            (int(self.x), int(self.y)),
            self.RADIUS
        )
