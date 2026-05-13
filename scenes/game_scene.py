import pygame

from core.game import Game
from core.assets import load_image

from scenes.menu_scene import MenuScene


class GameScene:

    def __init__(self, main_game, mode="survival"):

        self.main_game = main_game

        self.game = Game(mode=mode)

        self.font = pygame.font.SysFont(
            [
                "Orbitron",
                "Audiowide",
                "Eurostile",
                "Consolas"
            ],
            40
        )
        self.option_font = pygame.font.SysFont(
            [
                "Orbitron",
                "Audiowide",
                "Eurostile",
                "Consolas"
            ],
            32
        )
        self.game_over_options = [
            "Again",
            "Menu"
        ]
        self.selected_game_over_option = 0
        self.chose_icon = load_image(
            "chose.png",
            (36, 36)
        )

    def handle_event(self, event):

        if event.type == pygame.KEYDOWN:

            if self.game.running:

                if event.key == pygame.K_ESCAPE:

                    self.main_game.current_scene = MenuScene(
                        self.main_game
                    )

                if event.key == pygame.K_r:

                    self.game.reset()
            else:
                if event.key == pygame.K_UP:
                    self.selected_game_over_option = (
                        self.selected_game_over_option - 1
                    ) % len(self.game_over_options)

                elif event.key == pygame.K_DOWN:
                    self.selected_game_over_option = (
                        self.selected_game_over_option + 1
                    ) % len(self.game_over_options)

                elif event.key in (
                    pygame.K_RETURN,
                    pygame.K_KP_ENTER
                ):
                    if self.selected_game_over_option == 0:
                        self.game.reset()
                    else:
                        self.main_game.current_scene = MenuScene(
                            self.main_game
                        )

                elif event.key == pygame.K_ESCAPE:
                    self.main_game.current_scene = MenuScene(
                        self.main_game
                    )

    def update(self):

        if self.game.running:

            self.game.update()

    def draw_game_over(self, screen):

        overlay = pygame.Surface(
            (
                screen.get_width(),
                screen.get_height()
            )
        )

        overlay.set_alpha(180)

        overlay.fill((0, 0, 0))

        screen.blit(overlay, (0, 0))

        end_title = "VICTORY" if self.game.is_victory else "GAME OVER"
        end_color = (120, 255, 120) if self.game.is_victory else (255, 50, 50)
        game_over_text = self.font.render(
            end_title,
            True,
            end_color
        )

        screen.blit(
            game_over_text,
            game_over_text.get_rect(
                center=(screen.get_width() // 2, 200)
            )
        )

        y = 290

        for idx, option in enumerate(self.game_over_options):
            color = (255, 220, 120) if idx == self.selected_game_over_option else (255, 255, 255)
            option_text = self.option_font.render(
                option,
                True,
                color
            )
            option_rect = option_text.get_rect(
                center=(screen.get_width() // 2, y)
            )
            screen.blit(option_text, option_rect)

            if idx == self.selected_game_over_option:
                if self.chose_icon is not None:
                    screen.blit(
                        self.chose_icon,
                        (
                            option_rect.left - 46,
                            option_rect.centery - 18
                        )
                    )
                else:
                    marker = self.option_font.render(
                        ">",
                        True,
                        (255, 220, 120)
                    )
                    screen.blit(
                        marker,
                        (option_rect.left - 24, option_rect.top)
                    )
            y += 60

    def draw(self, screen):

        self.game.draw(screen)

        if not self.game.running:

            self.draw_game_over(screen)
