import pygame

from config.settings import TEXT_COLOR
from config.settings import HP_COLOR
from config.settings import INFO_PANEL_HEIGHT
from core.assets import load_image


class UI:

    def __init__(self):

        self.font = pygame.font.SysFont(
            [
                "Orbitron",
                "Audiowide",
                "Eurostile",
                "Consolas"
            ],
            32
        )

        self.player_hp_icon = load_image(
            "player_hp.png",
            (30, 30)
        )
        self.boss_hp_frame = load_image(
            "boss_hp.png",
            (300, 44)
        )

    def draw_timer(
        self,
        screen,
        survival_time
    ):

        text = self.font.render(
            f"{survival_time}",
            True,
            TEXT_COLOR
        )

        screen.blit(text, (20, 20))

    def draw_health(
        self,
        screen,
        health
    ):

        y = 20

        text = self.font.render(
            f"{health}",
            True,
            HP_COLOR
        )

        x = screen.get_width() - text.get_width() - 20

        if self.player_hp_icon is not None:
            x -= 40
            screen.blit(
                self.player_hp_icon,
                (x, y)
            )
            x += 40

        screen.blit(text, (x, y))

    def draw_boss_health(
        self,
        screen,
        health,
        max_health
    ):

        frame_width = 300
        frame_height = 44
        frame_x = (screen.get_width() - frame_width) // 2
        frame_y = (INFO_PANEL_HEIGHT - frame_height) // 2

        ratio = 0 if max_health <= 0 else max(0, min(1, health / max_health))
        fill_margin_x = 16
        fill_margin_y = 12
        full_fill_area_width = frame_width - fill_margin_x * 2
        shifted_fill_area_width = int(full_fill_area_width * 0.75)
        shifted_fill_x = frame_x + fill_margin_x + (full_fill_area_width - shifted_fill_area_width)
        fill_width = int(shifted_fill_area_width * ratio)
        fill_height = frame_height - fill_margin_y * 2

        pygame.draw.rect(
            screen,
            (60, 0, 0),
            (
                shifted_fill_x,
                frame_y + fill_margin_y,
                shifted_fill_area_width,
                fill_height
            ),
            border_radius=6
        )

        if fill_width > 0:
            pygame.draw.rect(
                screen,
                (255, 80, 80),
                (
                    shifted_fill_x,
                    frame_y + fill_margin_y,
                    fill_width,
                    fill_height
                ),
                border_radius=6
            )

        if self.boss_hp_frame is not None:
            screen.blit(
                self.boss_hp_frame,
                (frame_x, frame_y)
            )
