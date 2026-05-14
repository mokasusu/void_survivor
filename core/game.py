import pygame

from entities.player import Player
from entities.boss import Boss

from managers.bullet_manager import BulletManager
from managers.player_bullet_manager import PlayerBulletManager

from controllers.human_controller import HumanController

from core.collision import CollisionSystem
from core.ui import UI

from config.settings import WIDTH
from config.settings import HEIGHT
from config.settings import BACKGROUND_COLOR
from config.settings import INFO_PANEL_HEIGHT
from core.assets import load_image


class Game:

    def __init__(self, mode="survival"):

        self.mode = mode

        self.player = Player(
            700,
            200
        )
        self.boss = Boss() if self.mode == "boss" else None

        self.controller = HumanController()

        self.bullet_manager = BulletManager()
        self.player_bullet_manager = PlayerBulletManager()

        self.ui = UI()

        self.spawn_timer = 0
        self.shoot_cooldown = 0

        self.running = True
        self.is_victory = False

        self.start_time = pygame.time.get_ticks()
        self.end_time = None
        self.background = load_image(
            "background.png",
            (WIDTH, HEIGHT)
        )
        self.hit_effect_sprite = load_image(
            "hit.png",
            (48, 48)
        )
        self.hit_effects = []

    def draw_info_gradient(self, screen):

        panel = pygame.Surface(
            (WIDTH, INFO_PANEL_HEIGHT),
            pygame.SRCALPHA
        )

        for y in range(INFO_PANEL_HEIGHT):
            alpha = max(
                0,
                170 - int((170 * y) / INFO_PANEL_HEIGHT)
            )
            pygame.draw.line(
                panel,
                (8, 20, 40, alpha),
                (0, y),
                (WIDTH, y)
            )

        screen.blit(panel, (0, 0))

    def get_survival_time(self):

        if self.running or self.end_time is None:
            current_time = pygame.time.get_ticks()
        else:
            current_time = self.end_time

        total_seconds = (
            current_time - self.start_time
        ) // 1000

        minutes = total_seconds // 60

        seconds = total_seconds % 60

        return f"{minutes:02}:{seconds:02}"

    def get_elapsed_seconds(self):

        if self.running or self.end_time is None:
            current_time = pygame.time.get_ticks()
        else:
            current_time = self.end_time

        return max(
            0,
            (current_time - self.start_time) // 1000
        )

    def handle_collision(self):

        now = pygame.time.get_ticks()

        for bullet in self.bullet_manager.bullets[:]:

            if CollisionSystem.check(
                self.player,
                bullet
            ):

                self.bullet_manager.bullets.remove(
                    bullet
                )

                took_damage = self.player.take_damage(now)

                if not took_damage:
                    continue

                if self.player.is_dead():

                    self.running = False
                    self.is_victory = False
                    self.end_time = now

    def handle_boss_collision(self):

        if self.boss is None:
            return

        boss_rect = self.boss.get_rect()
        now = pygame.time.get_ticks()

        for bullet in self.player_bullet_manager.bullets[:]:
            if boss_rect.collidepoint(bullet.x, bullet.y):
                self.player_bullet_manager.bullets.remove(
                    bullet
                )
                self.boss.take_damage(1)
                self.boss.randomize_vertical_position()
                self.hit_effects.append(
                    (bullet.x, bullet.y, now + 100)
                )
                if self.boss.health <= 0:
                    self.running = False
                    self.is_victory = True
                    self.end_time = now
                    break

    def handle_player_boss_contact(self):

        if self.boss is None:
            return

        player_rect = pygame.Rect(
            self.player.x,
            self.player.y,
            self.player.WIDTH,
            self.player.HEIGHT
        )
        boss_rect = self.boss.get_rect()

        if not player_rect.colliderect(boss_rect):
            return

        now = pygame.time.get_ticks()
        took_damage = self.player.take_damage(now)
        if not took_damage:
            return

        if self.player.is_dead():
            self.running = False
            self.is_victory = False
            self.end_time = now

    def update_hit_effects(self):

        now = pygame.time.get_ticks()
        self.hit_effects = [
            effect
            for effect in self.hit_effects
            if effect[2] > now
        ]

    def update(self):

        action = self.controller.get_action()

        self.player.update(action)

        self.bullet_manager.update()
        self.player_bullet_manager.update()

        elapsed_seconds = self.get_elapsed_seconds()
        difficulty_scale = min(10, 1 + (elapsed_seconds // 20))
        spawn_interval = max(8, 40 - difficulty_scale * 2)
        bullets_per_wave = min(10, 1 + (difficulty_scale // 2))

        self.spawn_timer += 1

        if self.spawn_timer >= spawn_interval:

            for _ in range(bullets_per_wave):
                self.bullet_manager.spawn_bullet()

            self.spawn_timer = 0

        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1

        if self.controller.is_shooting() and self.shoot_cooldown == 0:
            shoot_x, shoot_y = self.player.get_shoot_origin()
            self.player_bullet_manager.shoot(shoot_x, shoot_y)
            self.shoot_cooldown = 16

        self.handle_boss_collision()
        self.handle_player_boss_contact()
        self.update_hit_effects()
        self.handle_collision()

    def draw(self, screen):

        if self.background is not None:
            screen.blit(self.background, (0, 0))
        else:
            screen.fill(
                BACKGROUND_COLOR
            )

        self.player.draw(screen)
        if self.boss is not None:
            self.boss.draw(screen)

        self.bullet_manager.draw(screen)
        self.player_bullet_manager.draw(screen)
        for x, y, _ in self.hit_effects:
            if self.hit_effect_sprite is not None:
                screen.blit(
                    self.hit_effect_sprite,
                    (
                        int(x - 24),
                        int(y - 24)
                    )
                )

        self.draw_info_gradient(screen)

        self.ui.draw_timer(
            screen,
            self.get_survival_time()
        )

        self.ui.draw_health(
            screen,
            self.player.health
        )
        if self.boss is not None:
            self.ui.draw_boss_health(
                screen,
                self.boss.health,
                self.boss.max_health
            )

    def reset(self):

        self.player = Player(
            700,
            200
        )
        self.boss = Boss() if self.mode == "boss" else None

        self.bullet_manager = BulletManager()
        self.player_bullet_manager = PlayerBulletManager()

        self.spawn_timer = 0
        self.shoot_cooldown = 0

        self.running = True
        self.is_victory = False

        self.start_time = pygame.time.get_ticks()
        self.end_time = None
        self.hit_effects = []
