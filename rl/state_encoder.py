class StateEncoder:

    def __init__(self, max_bullets=5):
        self.max_bullets = max_bullets

    def encode(self, game):
        return game.get_observation(max_bullets=self.max_bullets)
