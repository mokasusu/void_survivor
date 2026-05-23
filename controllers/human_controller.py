import pygame

from controllers.action import Action


class HumanController:

    def get_action(self):

        keys = pygame.key.get_pressed()

        up = keys[pygame.K_UP] or keys[pygame.K_w]
        down = keys[pygame.K_DOWN] or keys[pygame.K_s]
        left = keys[pygame.K_LEFT] or keys[pygame.K_a]
        right = keys[pygame.K_RIGHT] or keys[pygame.K_d]

        if up and left:

            return Action.UP_LEFT

        if up and right:

            return Action.UP_RIGHT

        if down and left:

            return Action.DOWN_LEFT

        if down and right:

            return Action.DOWN_RIGHT

        if up:

            return Action.UP

        if down:

            return Action.DOWN

        if left:

            return Action.LEFT

        if right:

            return Action.RIGHT

        return Action.IDLE

    def is_shooting(self):

        keys = pygame.key.get_pressed()

        return keys[pygame.K_SPACE]
