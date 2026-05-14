import pygame

from scenes.menu_scene import MenuScene
from config.settings import WIDTH
from config.settings import HEIGHT
from core.assets import load_image


class GuideScene:

    def __init__(self, game):

        self.game = game

        self.font = pygame.font.SysFont(
            "Arial",
            32
        )
        self.background = load_image(
            "background.png",
            (WIDTH, HEIGHT)
        )

    def handle_event(self, event):

        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_ESCAPE:

                self.game.current_scene = MenuScene(
                    self.game
                )

    def update(self):
        pass

    def draw(self, screen):

        if self.background is not None:
            screen.blit(self.background, (0, 0))
        else:
            screen.fill((30, 30, 30))

        texts = [
            "GUIDE",
            "",
            "Use Arrow Keys or W/A/S/D to move",
            "Use SPACE to shoot",
            "",
            "Press ESC to go back"
        ]

        y = 100

        for line in texts:

            text = self.font.render(
                line,
                True,
                (255, 255, 255)
            )

            screen.blit(text, (80, y))

            y += 50
