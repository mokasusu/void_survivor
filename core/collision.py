class CollisionSystem:

    @staticmethod
    def check(player, bullet):

        player_left = player.x
        player_right = player.x + player.WIDTH

        player_top = player.y
        player_bottom = player.y + player.HEIGHT

        bullet_left = bullet.x - bullet.RADIUS
        bullet_right = bullet.x + bullet.RADIUS

        bullet_top = bullet.y - bullet.RADIUS
        bullet_bottom = bullet.y + bullet.RADIUS

        return (
            player_left < bullet_right and
            player_right > bullet_left and
            player_top < bullet_bottom and
            player_bottom > bullet_top
        )