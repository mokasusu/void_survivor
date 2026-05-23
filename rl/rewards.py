import math


class RewardShaper:

    def __init__(self):
        self.prev_health = None
        self.prev_boss_health = None
        self.prev_min_dist = None

    def reset(self, game):
        self.prev_health = game.player.health
        self.prev_boss_health = (
            game.boss.health if game.boss else None
        )
        self.prev_min_dist = None

    def compute(self, game):
        reward = 0.0
        breakdown = {
            "survival": 0.0,
            "time_pressure": 0.0,
            "danger": 0.0,
            "damage_penalty": 0.0,
            "boss_damage": 0.0,
            "win_loss": 0.0,
        }

        is_survival = game.mode == "survival"

        health_loss = 0

        # =========================
        # Survival reward (time alive)
        # =========================
        if is_survival and game.running:
            reward += 0.02
            breakdown["survival"] += 0.02

        # =========================
        # Damage taken penalty
        # =========================
        if self.prev_health is not None:

            health_loss = (
                self.prev_health - game.player.health
            )

            if health_loss > 0:
                penalty = health_loss * 2.0
                reward -= penalty
                breakdown["damage_penalty"] -= penalty

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

            # reward escaping danger (moving away from closest bullet)
            if min_dist < 120 and health_loss == 0:
                if self.prev_min_dist is not None and min_dist > self.prev_min_dist:
                    gain = 0.02 if is_survival else 0.02
                    reward += gain
                    breakdown["danger"] += gain

        # =========================
        # Boss damage reward
        # =========================
        if (
            not is_survival
            and game.boss is not None
            and self.prev_boss_health is not None
        ):

            boss_damage = (
                self.prev_boss_health
                - game.boss.health
            )

            if boss_damage > 0:
                gain = boss_damage * 4.0
                reward += gain
                breakdown["boss_damage"] += gain

        # =========================
        # Win / lose
        # =========================
        if not game.running:

            if game.is_victory:
                reward += 50.0
                breakdown["win_loss"] += 50.0
            else:
                reward -= 20.0
                breakdown["win_loss"] -= 20.0

        # =========================
        # Update
        # =========================
        self.prev_health = game.player.health

        self.prev_boss_health = (
            game.boss.health if game.boss else None
        )
        self.prev_min_dist = min_dist if bullets else None

        return reward, breakdown