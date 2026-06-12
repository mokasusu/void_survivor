import pygame
import math

from entities.player import Player
from entities.boss import Boss
from entities.item import Item
from entities.black_hole import BlackHole

from managers.bullet_manager import BulletManager
from managers.player_bullet_manager import PlayerBulletManager

from controllers.human_controller import HumanController

from core.collision import CollisionSystem
from core.ui import UI

from config.settings import WIDTH, HEIGHT, BACKGROUND_COLOR, INFO_PANEL_HEIGHT
from core.assets import load_image


class Game:
    def __init__(self, mode="survival", controller=None, use_sim_time=False, sim_step_ms=16):
        self.mode = mode
        self.player = Player(700, 200)
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
        self.background = load_image("background.png", (WIDTH, HEIGHT))
        self.hit_effect_sprite = load_image("hit.png", (48, 48))
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

        self.items = []
        self.boss_last_health = 100

        # Floating text
        self.floating_texts = []

        # Screen flash
        self.screen_flash = None

        # Black hole
        self.black_hole = None
        self.void_phase_triggered = False

        # Enrage
        self.enrage_triggered = False
        self.enrage_announce_until = 0

        # Screen shake
        self.shake_until = 0
        self.shake_strength = 0

    # ── Helpers ─────────────────────────────────────────────
    def draw_info_gradient(self, screen):
        panel = pygame.Surface((WIDTH, INFO_PANEL_HEIGHT), pygame.SRCALPHA)
        for y in range(INFO_PANEL_HEIGHT):
            alpha = max(0, 170 - int((170 * y) / INFO_PANEL_HEIGHT))
            pygame.draw.line(panel, (8, 20, 40, alpha), (0, y), (WIDTH, y))
        screen.blit(panel, (0, 0))

    def get_survival_time(self):
        if self.running or self.end_time is None:
            current_time = self._now()
        else:
            current_time = self.end_time
        total_seconds = (current_time - self.start_time) // 1000
        return f"{total_seconds // 60:02}:{total_seconds % 60:02}"

    def get_elapsed_seconds(self):
        if self.running or self.end_time is None:
            current_time = self._now()
        else:
            current_time = self.end_time
        return max(0, (current_time - self.start_time) // 1000)

    def spawn_floating_text(self, text, x, y, color):
        now = pygame.time.get_ticks()
        self.floating_texts.append({"text": text, "x": x, "y": y,
                                    "color": color, "start": now, "duration": 1200})

    def trigger_screen_flash(self, color, duration_ms=300):
        self.screen_flash = {"color": color, "end": pygame.time.get_ticks() + duration_ms,
                             "duration": duration_ms}

    def trigger_shake(self, duration_ms=400, strength=6):
        self.shake_until = pygame.time.get_ticks() + duration_ms
        self.shake_strength = strength

    # ── Collision handlers ───────────────────────────────────
    def handle_collision(self):
        now = self._now()
        for bullet in self.bullet_manager.bullets[:]:
            if CollisionSystem.check(self.player, bullet):
                self.bullet_manager.bullets.remove(bullet)
                self.player_hit_count += 1
                took = self.player.take_damage(now)
                if took and self.player.is_dead():
                    self.running = False
                    self.is_victory = False
                    self.end_time = now

    def handle_boss_collision(self):
        if self.boss is None:
            return
        now = self._now()

        # Đạn người chơi vs Minion
        for minion in self.boss.minions[:]:
            minion_rect = minion.get_rect()
            for bullet in self.player_bullet_manager.bullets[:]:
                if minion_rect.collidepoint(bullet.x, bullet.y):
                    self.player_bullet_manager.bullets.remove(bullet)
                    minion.take_damage(1)
                    self.hit_effects.append((bullet.x, bullet.y, now + 100))

        # Đạn người chơi vs Boss
        boss_rect = self.boss.get_rect()
        for bullet in self.player_bullet_manager.bullets[:]:
            if boss_rect.collidepoint(bullet.x, bullet.y):
                self.player_bullet_manager.bullets.remove(bullet)
                self.boss.take_damage(1)
                self.hit_effects.append((bullet.x, bullet.y, now + 100))
                if self.boss.health <= 0:
                    self.running = False
                    self.is_victory = True
                    self.end_time = now
                    break

    def handle_player_boss_contact(self):
        if self.boss is None:
            return
        player_rect = pygame.Rect(self.player.x, self.player.y, self.player.WIDTH, self.player.HEIGHT)
        now = self._now()
        collision = player_rect.colliderect(self.boss.get_rect())
        for laser in self.boss.lasers:
            laser_rect = laser.get_damage_rect()
            if laser_rect and player_rect.colliderect(laser_rect):
                collision = True
        if not collision:
            return
        took = self.player.take_damage(now)
        if took and self.player.is_dead():
            self.running = False
            self.is_victory = False
            self.end_time = now

    def handle_item_collision(self):
        player_rect = pygame.Rect(self.player.x, self.player.y, self.player.WIDTH, self.player.HEIGHT)
        px = self.player.x + self.player.WIDTH // 2
        py = self.player.y
        for item in self.items[:]:
            if player_rect.colliderect(item.get_rect()):
                if item.type == "heart":
                    self.player.health = min(self.player.max_health, self.player.health + 1)
                    self.spawn_floating_text("+1 HP", px, py - 10, (255, 80, 80))
                    self.trigger_screen_flash((255, 50, 50), 250)
                elif item.type == "weapon":
                    self.player.upgrade_weapon()
                    self.spawn_floating_text(f"WEAPON UP!  LV.{self.player.weapon_level}",
                                            px, py - 10, (50, 230, 100))
                    self.trigger_screen_flash((50, 200, 80), 300)
                self.items.remove(item)

    def handle_black_hole(self):
        if self.black_hole is None:
            return
        self.black_hole.update()
        px = self.player.x + self.player.WIDTH // 2
        py = self.player.y + self.player.HEIGHT // 2
        fx, fy = self.black_hole.get_pull(px, py)
        self.player.x += fx
        self.player.y += fy
        self.player.clamp()

        # Nếu rơi vào lõi thì nhận sát thương
        dx = self.black_hole.x - px
        dy = self.black_hole.y - py
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < self.black_hole.inner_radius + 8:
            now = pygame.time.get_ticks()
            took = self.player.take_damage(now)
            if took and self.player.is_dead():
                self.running = False
                self.is_victory = False
                self.end_time = now

    # ── Phase triggers ───────────────────────────────────────
    def check_boss_phases(self):
        if self.boss is None:
            return
        hp = self.boss.health

        # Void phase: 30-35% HP
        if not self.void_phase_triggered and hp <= 35:
            self.void_phase_triggered = True
            self.boss.enter_void_phase()
            self.black_hole = BlackHole(WIDTH // 2, (INFO_PANEL_HEIGHT + HEIGHT) // 2)
            self.spawn_floating_text("VOID CORE ACTIVATED!", WIDTH // 2, HEIGHT // 2 - 60, (180, 50, 255))
            self.trigger_screen_flash((80, 0, 160), 500)
            self.trigger_shake(600, 8)

        # Enrage phase: 15% HP
        if not self.enrage_triggered and hp <= 15:
            self.enrage_triggered = True
            self.boss.enter_enrage()
            self.enrage_announce_until = pygame.time.get_ticks() + 2500
            self.spawn_floating_text("BOSS ENRAGED!", WIDTH // 2, HEIGHT // 2 - 80, (255, 30, 30))
            self.trigger_screen_flash((255, 30, 30), 600)
            self.trigger_shake(700, 10)

    # ── Update / Draw helpers ────────────────────────────────
    def update_hit_effects(self):
        now = self._now()
        self.hit_effects = [e for e in self.hit_effects if e[2] > now]

    def update_floating_texts(self):
        now = self._now()
        self.floating_texts = [t for t in self.floating_texts if now - t["start"] < t["duration"]]

    def update(self):
        action = self.controller.get_action()
        self.update_with_action(action, self.controller.is_shooting())

    def update_with_action(self, action, is_shooting):
        self.advance_time()
        self.player.update(action)
        self.bullet_manager.update()
        self.player_bullet_manager.update()

        elapsed = self.get_elapsed_seconds()
        difficulty_scale = min(10, 1 + elapsed // 20)
        spawn_interval = max(8, 40 - difficulty_scale * 2)
        bullets_per_wave = min(10, 1 + difficulty_scale // 2)

        self.spawn_timer += 1
        if self.spawn_timer >= spawn_interval:
            for _ in range(bullets_per_wave):
                self.bullet_manager.spawn_bullet()
            self.spawn_timer = 0

        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
        if is_shooting and self.shoot_cooldown == 0:
            sx, sy = self.player.get_shoot_origin()
            self.player_bullet_manager.shoot(sx, sy, self.player.weapon_level)
            self.shoot_cooldown = 16

        # Boss update
        if self.boss is not None:
            prev_hp = self.boss_last_health

            # Rớt đồ theo mốc máu
            if self.boss.health < prev_hp:
                for threshold in [75, 50, 25]:
                    if prev_hp >= threshold and self.boss.health < threshold:
                        self.items.append(Item(
                            self.boss.x, self.boss.y + self.boss.display_height // 2, "heart"))
                for threshold in [60, 30]:
                    if prev_hp >= threshold and self.boss.health < threshold:
                        self.items.append(Item(
                            self.boss.x, self.boss.y + self.boss.display_height // 2, "weapon"))
                self.boss_last_health = self.boss.health

            px = self.player.x + self.player.WIDTH // 2
            py = self.player.y + self.player.HEIGHT // 2
            self.boss.update(player_x=px, player_y=py)

            if self.boss.new_bullets:
                self.bullet_manager.bullets.extend(self.boss.new_bullets)
                self.boss.new_bullets = []

            self.check_boss_phases()

        # Black hole pull
        self.handle_black_hole()

        # Items
        for item in self.items:
            item.update()
        self.items = [item for item in self.items if not item.is_outside_screen()]

        self.handle_boss_collision()
        self.handle_player_boss_contact()
        self.handle_item_collision()
        self.update_hit_effects()
        self.update_floating_texts()
        self.handle_collision()

    def draw_floating_texts(self, screen):
        now = pygame.time.get_ticks()
        font = pygame.font.SysFont(["Orbitron", "Audiowide", "Consolas"], 22)
        for t in self.floating_texts:
            elapsed = now - t["start"]
            progress = elapsed / t["duration"]
            alpha = int(255 * (1.0 - progress))
            rise = int(50 * progress)
            surf = font.render(t["text"], True, t["color"])
            surf.set_alpha(alpha)
            screen.blit(surf, (t["x"] - surf.get_width() // 2, t["y"] - rise))

    def draw_screen_flash(self, screen):
        if self.screen_flash is None:
            return
        now = pygame.time.get_ticks()
        if now > self.screen_flash["end"]:
            self.screen_flash = None
            return
        remaining = self.screen_flash["end"] - now
        alpha = int(90 * (remaining / self.screen_flash["duration"]))
        flash_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        r, g, b = self.screen_flash["color"]
        flash_surf.fill((r, g, b, alpha))
        screen.blit(flash_surf, (0, 0))

    def get_shake_offset(self):
        now = pygame.time.get_ticks()
        if now > self.shake_until:
            return 0, 0
        import random as _r
        s = self.shake_strength
        return _r.randint(-s, s), _r.randint(-s, s)

    def draw_enrage_banner(self, screen):
        now = pygame.time.get_ticks()
        if now > self.enrage_announce_until:
            return
        font = pygame.font.SysFont(["Orbitron", "Audiowide", "Consolas"], 48)
        blink = (now // 200) % 2
        color = (255, 50, 50) if blink == 0 else (255, 150, 50)
        text = font.render("⚠  BOSS ENRAGED  ⚠", True, color)
        surf = pygame.Surface((text.get_width() + 40, text.get_height() + 16), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 140))
        screen.blit(surf, (WIDTH // 2 - surf.get_width() // 2, HEIGHT // 2 - 70))
        screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - 62))
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
        ox, oy = self.get_shake_offset()

        if self.background is not None:
            screen.blit(self.background, (ox, oy))
        else:
            screen.fill(BACKGROUND_COLOR)

        for item in self.items:
            item.draw(screen)

        if self.black_hole is not None:
            self.black_hole.draw(screen)

        self.player.draw(screen)

        if self.boss is not None:
            self.boss.draw(screen)

        self.bullet_manager.draw(screen)
        self.player_bullet_manager.draw(screen)

        for x, y, _ in self.hit_effects:
            if self.hit_effect_sprite is not None:
                screen.blit(self.hit_effect_sprite, (int(x - 24), int(y - 24)))

        self.draw_screen_flash(screen)
        self.draw_floating_texts(screen)
        self.draw_enrage_banner(screen)

        self.draw_info_gradient(screen)
        self.ui.draw_timer(screen, self.get_survival_time())
        self.ui.draw_health(screen, self.player.health)
        self.ui.draw_weapon_level(screen, self.player.weapon_level)

        if self.boss is not None:
            self.ui.draw_boss_health(screen, self.boss.health, self.boss.max_health)

        self.ui.draw_episode_label(
            screen,
            self.episode_label
        )

    def set_episode_label(self, label):

        self.episode_label = label

    def reset(self):
        self.player = Player(700, 200)
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
        self.items = []
        self.boss_last_health = 100
        self.floating_texts = []
        self.screen_flash = None
        self.black_hole = None
        self.void_phase_triggered = False
        self.enrage_triggered = False
        self.enrage_announce_until = 0
        self.shake_until = 0
        self.shake_strength = 0

    def _now(self):
        if self.use_sim_time:
            return self.time_ms
        return pygame.time.get_ticks()

    def advance_time(self, steps=1):
        if not self.use_sim_time:
            return
        self.time_ms += self.sim_step_ms * steps
