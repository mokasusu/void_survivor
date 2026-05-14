class RewardShaper:

    def __init__(self):
        self.prev_health = None
        self.prev_boss_health = None

    def reset(self, game):
        self.prev_health = game.player.health
        self.prev_boss_health = game.boss.health if game.boss else None

    def compute(self, game):
        reward = 0.01

        if self.prev_health is not None and game.player.health < self.prev_health:
            reward -= 1.0

        if game.boss is not None and self.prev_boss_health is not None:
            if game.boss.health < self.prev_boss_health:
                reward += 0.2

        if not game.running:
            reward += 5.0 if game.is_victory else -5.0

        self.prev_health = game.player.health
        self.prev_boss_health = game.boss.health if game.boss else None

        return reward
