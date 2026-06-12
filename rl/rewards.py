class RewardShaper:

    def __init__(self):
        self.prev_health = None
        self.prev_boss_health = None
        self.prev_bullet_x = {}

    def reset(self, game):
        self.prev_health = game.player.health
        self.prev_boss_health = (
            game.boss.health if game.boss else None
        )
        self.prev_bullet_x = {}

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

        health_loss = 0

        # =========================
        # Damage taken penalty
        # =========================
        if self.prev_health is not None:

            health_loss = (
                self.prev_health - game.player.health
            )

            if health_loss > 0:
                penalty = health_loss * 10.0
                reward -= penalty
                breakdown["damage_penalty"] -= penalty

        # =========================
        # Successful dodge reward
        # =========================
        bullets = game.bullet_manager.bullets

        if bullets:

            player_cx = game.player.x + game.player.WIDTH / 2
            player_cy = game.player.y + game.player.HEIGHT / 2

            dodge_reward = 0.0
            new_prev_bullet_x = {}

            for bullet in bullets:
                bullet_id = id(bullet)
                prev_x = self.prev_bullet_x.get(bullet_id)
                new_prev_bullet_x[bullet_id] = bullet.x

                if prev_x is None:
                    continue

                if prev_x < player_cx <= bullet.x and health_loss == 0:
                    y_dist = abs(bullet.y - player_cy)
                    if y_dist <= 80:
                        dodge_reward += 0.3

            if dodge_reward > 0:
                reward += dodge_reward
                breakdown["danger"] += dodge_reward

            self.prev_bullet_x = new_prev_bullet_x
        else:
            self.prev_bullet_x = {}

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
                gain = boss_damage * 5.0
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
                reward -= 40.0
                breakdown["win_loss"] -= 40.0

        # =========================
        # Update
        # =========================
        self.prev_health = game.player.health

        self.prev_boss_health = (
            game.boss.health if game.boss else None
        )

        return reward, breakdown