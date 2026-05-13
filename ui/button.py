import pygame
from config.settings import PLAYER_COLOR
from core.assets import load_image


class Button:

    def __init__(
        self,
        x,
        y,
        width,
        height,
        text,
        font
    ):

        self.rect = pygame.Rect(
            x,
            y,
            width,
            height
        )

        self.text = text

        self.font = font

        self.color = (70, 70, 70)

        self.hover_color = (120, 120, 120)

        self.text_color = (255, 255, 255)
        self.selected_text_color = (255, 235, 90)
        self.player_icon_size = 28
        self.player_icon = load_image(
            "chose.png",
            (45, 45)
        )

    def draw(self, screen, selected=False):

        mouse_pos = pygame.mouse.get_pos()
        is_active = self.rect.collidepoint(mouse_pos) or selected

        text_color = self.selected_text_color if is_active else self.text_color

        text_surface = self.font.render(
            self.text,
            True,
            text_color
        )

        text_rect = text_surface.get_rect(
            center=self.rect.center
        )

        screen.blit(
            text_surface,
            text_rect
        )

        if is_active:
            icon_x = text_rect.left - 48
            icon_y = text_rect.centery - (self.player_icon_size // 2)

            if self.player_icon is not None:
                screen.blit(
                    self.player_icon,
                    (icon_x, text_rect.centery - 20)
                )
            else:
                pygame.draw.rect(
                    screen,
                    PLAYER_COLOR,
                    (
                        icon_x,
                        icon_y,
                        self.player_icon_size,
                        self.player_icon_size
                    )
                )

    def is_clicked(self, event):

        if event.type == pygame.MOUSEBUTTONDOWN:

            if event.button == 1:

                return self.rect.collidepoint(
                    event.pos
                )

        return False
