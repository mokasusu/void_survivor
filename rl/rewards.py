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

        # =========================
        # Survival reward
        # =========================
        reward += 0.02

        # =========================
        # Bullet dodge reward
        # =========================
        bullets = game.bullet_manager.bullets if game.bullet_manager else []

        if bullets:
            player_cx = game.player.x + (game.player.WIDTH / 2)
            player_cy = game.player.y + (game.player.HEIGHT / 2)

            min_dist = float("inf")

            for bullet in bullets:
                dx = bullet.x - player_cx
                dy = bullet.y - player_cy
                dist_sq = dx * dx + dy * dy

                if dist_sq < min_dist:
                    min_dist = dist_sq

            if self.prev_min_bullet_dist is not None:
                # Reward moving away from nearby bullets
                dist_delta = min_dist - self.prev_min_bullet_dist

                near_threshold = 150.0
                near_threshold_sq = near_threshold * near_threshold

                if self.prev_min_bullet_dist < near_threshold_sq:
                    reward += dist_delta * 0.00005

            self.prev_min_bullet_dist = min_dist

        else:
            self.prev_min_bullet_dist = None

        # =========================
        # Damage taken penalty
        # =========================
        if self.prev_health is not None:
            health_loss = self.prev_health - game.player.health

            if health_loss > 0:
                reward -= health_loss * 3.0

        # =========================
        # Boss damage reward
        # =========================
        if game.boss is not None and self.prev_boss_health is not None:
            boss_damage = self.prev_boss_health - game.boss.health

            if boss_damage > 0:
                reward += boss_damage * 2.0

        # =========================
        # Win / lose reward
        # =========================
        if not game.running:
            if game.is_victory:
                reward += 30.0
            else:
                reward -= 10.0

        # =========================
        # Clamp reward (stability)
        # =========================
        reward = max(min(reward, 10.0), -10.0)

        # =========================
        # Update previous state
        # =========================
        self.prev_health = game.player.health
        self.prev_boss_health = (
            game.boss.health if game.boss else None
        )

        return reward