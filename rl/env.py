import os
import pygame

from core.game import Game
from controllers.action import Action
from config.settings import WIDTH
from config.settings import HEIGHT
from rl.rewards import RewardShaper
from rl.state_encoder import StateEncoder


class VoidSurvivorEnv:

    ACTIONS = {
        0: (Action.IDLE, False),
        1: (Action.UP, False),
        2: (Action.DOWN, False),
        3: (Action.LEFT, False),
        4: (Action.RIGHT, False),
        5: (Action.IDLE, True),
        6: (Action.UP, True),
        7: (Action.DOWN, True),
        8: (Action.LEFT, True),
        9: (Action.RIGHT, True),
    }

    def __init__(self, mode="survival", render=False, max_steps=5000, render_fps=60, encoder=None, reward_shaper=None):

        self.mode = mode
        self.render_enabled = render
        self.max_steps = max_steps
        self.render_fps = render_fps

        if not self.render_enabled:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        pygame.init()
        pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock() if self.render_enabled else None

        self.game = Game(mode=self.mode)
        self.steps = 0
        self.encoder = encoder or StateEncoder()
        self.reward_shaper = reward_shaper or RewardShaper()
        self.reward_shaper.reset(self.game)

    def reset(self):

        self.game.reset()
        self.steps = 0
        self.reward_shaper.reset(self.game)
        return self.encoder.encode(self.game)

    def set_episode_label(self, label):

        self.game.set_episode_label(label)

    def step(self, action_index):

        action, is_shooting = self.ACTIONS.get(action_index, (Action.IDLE, False))
        self.game.update_with_action(action, is_shooting)

        self.steps += 1
        done = not self.game.running or self.steps >= self.max_steps

        reward = self.reward_shaper.compute(self.game)

        observation = self.encoder.encode(self.game)
        info = {
            "survival_time": self.game.get_survival_time(),
            "elapsed_seconds": self.game.get_elapsed_seconds(),
            "is_victory": self.game.is_victory
        }

        if self.render_enabled:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.game.running = False

            self.game.draw(pygame.display.get_surface())
            pygame.display.flip()

            if self.clock:
                self.clock.tick(self.render_fps)

        return observation, reward, done, info

    def close(self):

        pygame.quit()
