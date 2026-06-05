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

    def __init__(self, mode="survival", controller=None, use_sim_time=False, sim_step_ms=16):

        self.mode = mode

        self.player = Player(
            700,
            200
        )
        self.boss = Boss() if self.mode == "boss" else None

        self.controller = controller or HumanController()

        self.bullet_manager = BulletManager()
        self.player_bullet_manager = PlayerBulletManager()

        self.ui = UI()

        self.spawn_timer = 0
        self.shoot_cooldown = 0

        self.running = True
        self.is_victory = False

        self.player_hit_count = 0

        self.use_sim_time = use_sim_time
        self.sim_step_ms = max(1, int(sim_step_ms))
        self.time_ms = pygame.time.get_ticks()
        self.start_time = self._now()
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
        self.episode_label = None

    def _now(self):

        if self.use_sim_time:
            return self.time_ms

        return pygame.time.get_ticks()

    def advance_time(self, steps=1):

        if not self.use_sim_time:
            return

        self.time_ms += self.sim_step_ms * max(1, int(steps))

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
            current_time = self._now()
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
            current_time = self._now()
        else:
            current_time = self.end_time

        return max(
            0,
            (current_time - self.start_time) // 1000
        )

    def handle_collision(self):

        now = self._now()

        for bullet in self.bullet_manager.bullets[:]:

            if CollisionSystem.check(
                self.player,
                bullet
            ):

                self.bullet_manager.bullets.remove(
                    bullet
                )

                self.player_hit_count += 1
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
        now = self._now()

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

        now = self._now()
        took_damage = self.player.take_damage(now)
        if not took_damage:
            return

        if self.player.is_dead():
            self.running = False
            self.is_victory = False
            self.end_time = now

    def update_hit_effects(self):

        now = self._now()
        self.hit_effects = [
            effect
            for effect in self.hit_effects
            if effect[2] > now
        ]

    def update(self):

        action = self.controller.get_action()

        self.update_with_action(
            action,
            self.controller.is_shooting()
        )

    def update_with_action(self, action, is_shooting):

        self.advance_time()

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

        if is_shooting and self.shoot_cooldown == 0:
            shoot_x, shoot_y = self.player.get_shoot_origin()
            self.player_bullet_manager.shoot(shoot_x, shoot_y)
            self.shoot_cooldown = 16

        self.handle_boss_collision()
        self.handle_player_boss_contact()
        self.update_hit_effects()
        self.handle_collision()

    def get_observation(self, max_bullets=5):

        bullets_list = self.bullet_manager.bullets
        player_cx = self.player.x + self.player.WIDTH / 2
        player_cy = self.player.y + self.player.HEIGHT / 2

        def _danger_score(bullet):
            if bullet.speed <= 0:
                return float("inf")
            if bullet.x >= player_cx:
                time_to_reach = float("inf")
            else:
                time_to_reach = (player_cx - bullet.x) / bullet.speed
            y_dist = abs(bullet.y - player_cy) / max(1, HEIGHT)
            return time_to_reach + y_dist * 0.5

        if max_bullets is None or max_bullets >= len(bullets_list):
            bullets = bullets_list
        else:
            bullets = sorted(
                bullets_list,
                key=_danger_score
            )
        features = []

        player_y = self.player.y / max(1, HEIGHT)
        features.append(player_y)

        for bullet in bullets[:max_bullets]:
            features.append(bullet.x / max(1, WIDTH))
            features.append(bullet.y / max(1, HEIGHT))
            features.append(bullet.speed / 12)
            features.append(bullet.RADIUS / max(1, WIDTH))

        missing = max_bullets - min(max_bullets, len(bullets))
        for _ in range(missing):
            features.extend([0.0, 0.0, 0.0, 0.0])

        if self.boss is not None:
            boss_x = self.boss.x / max(1, WIDTH)
            boss_y = self.boss.y / max(1, HEIGHT)
            boss_w = self.boss.display_width / max(1, WIDTH)
            boss_h = self.boss.display_height / max(1, HEIGHT)
            boss_ratio = 0 if self.boss.max_health <= 0 else self.boss.health / self.boss.max_health
            features.extend([boss_x, boss_y, boss_w, boss_h, boss_ratio])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        return features

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

        self.ui.draw_episode_label(
            screen,
            self.episode_label
        )

    def set_episode_label(self, label):

        self.episode_label = label

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

        self.player_hit_count = 0

        if self.use_sim_time:
            self.time_ms = 0
            self.start_time = 0
        else:
            self.start_time = pygame.time.get_ticks()
        self.end_time = None
        self.hit_effects = []
