import pygame

from scenes.menu_scene import MenuScene
from config.settings import PLAYER_COLOR
from config.settings import WIDTH
from config.settings import HEIGHT
from core.assets import load_image


class ModeScene:

    def __init__(self, game):

        self.game = game

        self.font = pygame.font.SysFont(
            "Arial",
            40
        )

        self.options = [
            "Survival",
            "Boss",
            "Back"
        ]

        if self.game.selected_mode == "boss":
            self.selected_index = 1
        else:
            self.selected_index = 0
        self.background = load_image(
            "background.png",
            (WIDTH, HEIGHT)
        )
        self.player_icon = load_image(
            "chose.png",
            (40, 40)
        )

    def _activate_selected(self):

        if self.selected_index == 0:
            self.game.selected_mode = "survival"
            self.game.current_scene = MenuScene(
                self.game
            )
        elif self.selected_index == 1:
            self.game.selected_mode = "boss"
            self.game.current_scene = MenuScene(
                self.game
            )
        else:
            self.game.current_scene = MenuScene(
                self.game
            )

    def handle_event(self, event):

        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_ESCAPE:

                self.game.current_scene = MenuScene(
                    self.game
                )

            elif event.key == pygame.K_UP:

                self.selected_index = (
                    self.selected_index - 1
                ) % len(self.options)

            elif event.key == pygame.K_DOWN:

                self.selected_index = (
                    self.selected_index + 1
                ) % len(self.options)

            elif event.key in (
                pygame.K_RETURN,
                pygame.K_KP_ENTER
            ):

                self._activate_selected()

    def update(self):
        pass

    def draw(self, screen):

        if self.background is not None:
            screen.blit(self.background, (0, 0))
        else:
            screen.fill((40, 20, 20))

        title = self.font.render(
            "MODE",
            True,
            (255, 255, 255)
        )

        title_rect = title.get_rect(
            center=(screen.get_width() // 2, 90)
        )
        screen.blit(title, title_rect)

        y = 220

        for idx, option in enumerate(self.options):

            color = (255, 220, 120) if idx == self.selected_index else (255, 255, 255)
            text = self.font.render(
                option,
                True,
                color
            )

            text_rect = text.get_rect(
                center=(screen.get_width() // 2, y)
            )
            screen.blit(text, text_rect)

            if idx == self.selected_index:
                if self.player_icon is not None:
                    screen.blit(
                        self.player_icon,
                        (
                            text_rect.left - 50,
                            text_rect.centery - 20
                        )
                    )
                else:
                    pygame.draw.rect(
                        screen,
                        PLAYER_COLOR,
                        (
                            text_rect.left - 28,
                            y + 12,
                            16,
                            16
                        )
                    )

            y += 70
