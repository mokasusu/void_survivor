from entities.player_bullet import PlayerBullet


class PlayerBulletManager:

    def __init__(self):

        self.bullets = []

    def shoot(self, x, y, weapon_level=1):

        if weapon_level == 1:
            self.bullets.append(PlayerBullet(x, y, -14, 0))
        elif weapon_level == 2:
            self.bullets.append(PlayerBullet(x, y - 10, -14, 0))
            self.bullets.append(PlayerBullet(x, y + 10, -14, 0))
        elif weapon_level >= 3:
            self.bullets.append(PlayerBullet(x, y, -14, 0))
            self.bullets.append(PlayerBullet(x, y, -14, -2))
            self.bullets.append(PlayerBullet(x, y, -14, 2))

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
