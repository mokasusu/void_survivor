import math


class RewardShaper:

    def __init__(self):
        self.prev_health = None
        self.prev_boss_health = None
        self.prev_min_bullet_dist = None

    def reset(self, game):
        self.prev_health = game.player.health
        self.prev_boss_health = game.boss.health if game.boss else None
        self.prev_min_bullet_dist = None

    def compute(self, game):
        reward = 0.0

        # sống mỗi frame
        reward += 0.01

        # dodge gần đạn
        bullets = game.bullet_manager.bullets if game.bullet_manager else []
        if bullets:
            player_cx = game.player.x + (game.player.WIDTH / 2)
            player_cy = game.player.y + (game.player.HEIGHT / 2)
            min_dist = None
            for bullet in bullets:
                dx = bullet.x - player_cx
                dy = bullet.y - player_cy
                dist = math.hypot(dx, dy)
                if min_dist is None or dist < min_dist:
                    min_dist = dist

            near_threshold = 120.0
            if (
                min_dist is not None
                and self.prev_min_bullet_dist is not None
                and self.prev_min_bullet_dist < near_threshold
                and min_dist > self.prev_min_bullet_dist + 5
            ):
                reward += 0.05

            self.prev_min_bullet_dist = min_dist
        else:
            self.prev_min_bullet_dist = None

        if self.prev_health is not None and game.player.health < self.prev_health:
            reward -= 1.0

        if game.boss is not None and self.prev_boss_health is not None:
            if game.boss.health < self.prev_boss_health:
                reward += 1.0

        if not game.running:
            reward += 10.0 if game.is_victory else -10.0

        self.prev_health = game.player.health
        self.prev_boss_health = game.boss.health if game.boss else None

        return reward
