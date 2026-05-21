import math


class RewardShaper:

    def __init__(self):
        self.prev_health = None
        self.prev_boss_health = None

    def reset(self, game):
        self.prev_health = game.player.health
        self.prev_boss_health = (
            game.boss.health if game.boss else None
        )

    def compute(self, game):

        reward = 0.0

        # =========================
        # Survival reward
        # =========================
        reward += 0.01

        # =========================
        # Small time pressure
        # =========================
        reward -= 0.001

        # =========================
        # Bullet danger reward
        # =========================
        bullets = game.bullet_manager.bullets

        if bullets:

            player_cx = game.player.x + game.player.WIDTH / 2
            player_cy = game.player.y + game.player.HEIGHT / 2

            min_dist = float("inf")

            for bullet in bullets:

                dx = bullet.x - player_cx
                dy = bullet.y - player_cy

                dist = math.sqrt(dx * dx + dy * dy)

                if dist < min_dist:
                    min_dist = dist

            # reward surviving in danger
            if min_dist < 120:
                reward += 0.02

        # =========================
        # Damage taken penalty
        # =========================
        if self.prev_health is not None:

            health_loss = (
                self.prev_health - game.player.health
            )

            if health_loss > 0:
                reward -= health_loss * 4.0

        # =========================
        # Boss damage reward
        # =========================
        if (
            game.boss is not None
            and self.prev_boss_health is not None
        ):

            boss_damage = (
                self.prev_boss_health
                - game.boss.health
            )

            if boss_damage > 0:
                reward += boss_damage * 2.0

        # =========================
        # Win / lose
        # =========================
        if not game.running:

            if game.is_victory:
                reward += 50.0
            else:
                reward -= 20.0

        # =========================
        # Clamp
        # =========================
        reward = max(min(reward, 10), -10)

        # =========================
        # Update
        # =========================
        self.prev_health = game.player.health

        self.prev_boss_health = (
            game.boss.health if game.boss else None
        )

        return reward