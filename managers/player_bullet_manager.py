from entities.player_bullet import PlayerBullet


class PlayerBulletManager:

    def __init__(self):

        self.bullets = []

    def shoot(self, x, y):

        bullet = PlayerBullet(
            x,
            y,
            -14
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
