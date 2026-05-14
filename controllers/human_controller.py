import pygame

from controllers.action import Action


class HumanController:

    def get_action(self):

        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP] or keys[pygame.K_w]:

            return Action.UP

        if keys[pygame.K_DOWN] or keys[pygame.K_s]:

            return Action.DOWN

        if keys[pygame.K_LEFT] or keys[pygame.K_a]:

            return Action.LEFT

        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:

            return Action.RIGHT

        return Action.IDLE

    def is_shooting(self):

        keys = pygame.key.get_pressed()

        return keys[pygame.K_SPACE]
