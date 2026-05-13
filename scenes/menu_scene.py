import pygame

from ui.button import Button
from config.settings import WIDTH
from config.settings import HEIGHT
from core.assets import load_image


class MenuScene:

    def __init__(self, game):

        self.game = game

        self.font = pygame.font.SysFont(
            "Arial",
            40
        )

        self.title_font = pygame.font.SysFont(
            "Arial",
            64
        )
        self.background = load_image(
            "background.png",
            (WIDTH, HEIGHT)
        )
        self.logo = load_image("logo.png")
        if self.logo is not None:
            logo_width, logo_height = self.logo.get_size()
            self.logo = pygame.transform.smoothscale(
                self.logo,
                (
                    int(logo_width / 1.5),
                    int(logo_height / 1.5)
                )
            )

        button_width = 200
        button_height = 60
        button_step = 70
        menu_center_y = int((HEIGHT / 3 + HEIGHT) / 2)
        first_button_y = menu_center_y - (
            (button_height * 4 + (button_step - button_height) * 3) // 2
        )
        button_x = (WIDTH - button_width) // 2

        self.play_button = Button(
            button_x,
            first_button_y,
            button_width,
            button_height,
            "Play",
            self.font
        )

        self.mode_button = Button(
            button_x,
            first_button_y + button_step,
            button_width,
            button_height,
            "Mode",
            self.font
        )

        self.guide_button = Button(
            button_x,
            first_button_y + button_step * 2,
            button_width,
            button_height,
            "Guide",
            self.font
        )

        self.exit_button = Button(
            button_x,
            first_button_y + button_step * 3,
            button_width,
            button_height,
            "Exit",
            self.font
        )

        self.buttons = [
            self.play_button,
            self.mode_button,
            self.guide_button,
            self.exit_button
        ]

        self.selected_index = 0

    def _activate_selected(self):

        if self.selected_index == 0:

            from scenes.game_scene import GameScene

            self.game.current_scene = GameScene(
                self.game,
                mode=self.game.selected_mode
            )

        elif self.selected_index == 1:

            from scenes.mode_scene import ModeScene

            self.game.current_scene = ModeScene(
                self.game
            )

        elif self.selected_index == 2:

            from scenes.guide_scene import GuideScene

            self.game.current_scene = GuideScene(
                self.game
            )

        elif self.selected_index == 3:

            self.game.running = False

    def handle_event(self, event):

        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_UP:

                self.selected_index = (
                    self.selected_index - 1
                ) % len(self.buttons)

            elif event.key == pygame.K_DOWN:

                self.selected_index = (
                    self.selected_index + 1
                ) % len(self.buttons)

            elif event.key in (
                pygame.K_RETURN,
                pygame.K_KP_ENTER
            ):

                self._activate_selected()
                return

        if self.play_button.is_clicked(event):

            self.selected_index = 0

            from scenes.game_scene import GameScene

            self.game.current_scene = GameScene(
                self.game,
                mode=self.game.selected_mode
            )

        elif self.mode_button.is_clicked(event):

            self.selected_index = 1

            from scenes.mode_scene import ModeScene

            self.game.current_scene = ModeScene(
                self.game
            )

        elif self.guide_button.is_clicked(event):

            self.selected_index = 2

            from scenes.guide_scene import GuideScene

            self.game.current_scene = GuideScene(
                self.game
            )

        elif self.exit_button.is_clicked(event):

            self.selected_index = 3

            self.game.running = False

    def update(self):
        mode_label = (
            "surviral"
            if self.game.selected_mode == "survival"
            else "boss"
        )
        self.mode_button.text = f"Mode: {mode_label}"

    def draw(self, screen):

        if self.background is not None:
            screen.blit(self.background, (0, 0))
        else:
            screen.fill((20, 20, 40))

        if self.logo is not None:
            logo_rect = self.logo.get_rect()
            header_height = HEIGHT // 3

            logo_rect.midbottom = (
                WIDTH // 2,
                header_height + 60
            )
            screen.blit(self.logo, logo_rect)

        for idx, button in enumerate(self.buttons):

            button.draw(
                screen,
                selected=(idx == self.selected_index)
            )
