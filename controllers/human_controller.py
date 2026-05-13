import pygame

from controllers.action import Action


class HumanController:

    def get_action(self):

        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP]:

            return Action.UP

        if keys[pygame.K_DOWN]:

            return Action.DOWN

        return Action.IDLE

    def is_shooting(self):

        keys = pygame.key.get_pressed()

        return keys[pygame.K_SPACE]
