"""
GamePPO — Game core dành riêng cho pipeline PPO / Curriculum Learning.

Khác biệt so với Game gốc:
  - Dùng BossPPO thay cho Boss (có stage behavior + shoot).
  - Dùng BossBulletManager cho đạn 2D của Boss.
  - Boss.update() được gọi mỗi frame và trả về đạn mới.
  - BulletManager gốc vẫn được giữ (survival mode) nhưng trong boss mode không dùng.
  - Có init_match(stage) để Curriculum reset từng trận.
"""

import pygame

from entities.player import Player
from entities.boss_ppo import BossPPO
from managers.bullet_manager import BulletManager
from managers.boss_bullet_manager import BossBulletManager
from managers.player_bullet_manager import PlayerBulletManager
from core.collision import CollisionSystem
from core.ui import UI
from core.assets import load_image
from config.settings import WIDTH, HEIGHT, BACKGROUND_COLOR, INFO_PANEL_HEIGHT

SHOOT_COOLDOWN_FRAMES = 6  # khớp với StateEncoder


class GamePPO:

    def __init__(self, use_sim_time: bool = True, sim_step_ms: int = 16):
        self.use_sim_time = use_sim_time
        self.sim_step_ms = max(1, sim_step_ms)
        self.time_ms = 0

        # Đảm bảo pygame đã init (headless nếu chưa có display)
        import os
        if not pygame.get_init():
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            pygame.init()
        if pygame.display.get_surface() is None:
            pygame.display.set_mode((WIDTH, HEIGHT))

        self.player = Player(700, 300)
        self.boss: BossPPO | None = None

        self.bullet_manager = BulletManager()           # survival bullets (Stage 2+ env bullets)
        self.boss_bullet_manager = BossBulletManager()  # đạn do Boss bắn ra
        self.player_bullet_manager = PlayerBulletManager()

        self.ui = UI()
        self.shoot_cooldown: int = 0

        self.running = True
        self.is_victory = False
        self.start_time = 0
        self.end_time: int | None = None
        self.hit_effects: list = []

        self.background = load_image("background.png", (WIDTH, HEIGHT))
        self.hit_effect_sprite = load_image("hit.png", (48, 48))

        self.episode_label: str | None = None
        self._active_stage = 1

        # PPO reward tracking
        self.damage_dealt_this_step: int = 0
        self.player_hit_this_step: bool = False

    # ------------------------------------------------------------------
    # Time
    # ------------------------------------------------------------------

    def _now(self) -> int:
        return self.time_ms if self.use_sim_time else pygame.time.get_ticks()

    def advance_time(self):
        if self.use_sim_time:
            self.time_ms += self.sim_step_ms

    # ------------------------------------------------------------------
    # Curriculum — khởi tạo trận đấu theo Stage
    # ------------------------------------------------------------------

    def init_match(self, stage: int, stage_steps: int = 0):
        """Khởi tạo lại toàn bộ trạng thái game cho một trận đấu mới."""
        self._active_stage = stage

        self.player = Player(700, 300)
        self.boss = BossPPO(stage=stage, stage_steps=stage_steps)

        self.bullet_manager = BulletManager()
        self.boss_bullet_manager = BossBulletManager()
        self.player_bullet_manager = PlayerBulletManager()

        self.shoot_cooldown = 0
        self.running = True
        self.is_victory = False

        if self.use_sim_time:
            self.time_ms = 0
            self.start_time = 0
        else:
            self.start_time = pygame.time.get_ticks()

        self.end_time = None
        self.hit_effects = []

        self.damage_dealt_this_step = 0
        self.player_hit_this_step = False

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_with_action(self, action_enum, is_shooting: bool):
        """Nhận action từ RL agent và cập nhật game 1 frame."""
        self.advance_time()
        now = self._now()

        # Reset per-step counters
        self.damage_dealt_this_step = 0
        self.player_hit_this_step = False

        # --- Player movement ---
        self.player.update(action_enum)

        # --- Boss update → trả về đạn mới ---
        if self.boss is not None:
            agent_cx = self.player.x + self.player.WIDTH / 2
            agent_cy = self.player.y + self.player.HEIGHT / 2
            new_bullets = self.boss.update(agent_cx, agent_cy)
            self.boss_bullet_manager.spawn_from_dicts(new_bullets)

        # --- Cập nhật đạn ---
        if self.boss is not None:
            agent_cx = self.player.x + self.player.WIDTH / 2
            agent_cy = self.player.y + self.player.HEIGHT / 2
            self.boss_bullet_manager.update(agent_cx, agent_cy)
        self.bullet_manager.update()
        self.player_bullet_manager.update()

        # --- Bắn ---
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
        if is_shooting and self.shoot_cooldown == 0:
            sx, sy = self.player.get_shoot_origin()
            self.player_bullet_manager.shoot(sx, sy)
            self.shoot_cooldown = SHOOT_COOLDOWN_FRAMES

        # --- Collision ---
        self._handle_boss_bullet_collision(now)
        self._handle_boss_player_bullet_collision(now)
        self._handle_player_boss_contact(now)

        self._update_hit_effects()

    # ------------------------------------------------------------------
    # Collision helpers
    # ------------------------------------------------------------------

    def _handle_boss_bullet_collision(self, now: int):
        """Đạn Boss trúng Player."""
        player_rect = pygame.Rect(
            self.player.x, self.player.y,
            self.player.WIDTH, self.player.HEIGHT
        )
        for b in self.boss_bullet_manager.bullets[:]:
            if player_rect.collidepoint(b.x, b.y):
                self.boss_bullet_manager.bullets.remove(b)
                took = self.player.take_damage(now)
                if took:
                    self.player_hit_this_step = True
                    if self.player.is_dead():
                        self.running = False
                        self.is_victory = False
                        self.end_time = now
                        return

    def _handle_boss_player_bullet_collision(self, now: int):
        """Đạn Player trúng Boss."""
        if self.boss is None:
            return
        boss_rect = self.boss.get_rect()
        for b in self.player_bullet_manager.bullets[:]:
            if boss_rect.collidepoint(b.x, b.y):
                self.player_bullet_manager.bullets.remove(b)
                self.boss.take_damage(1)
                self.damage_dealt_this_step += 1
                self.hit_effects.append((b.x, b.y, now + 100))
                if self.boss.health <= 0:
                    self.running = False
                    self.is_victory = True
                    self.end_time = now
                    return

    def _handle_player_boss_contact(self, now: int):
        """Va chạm trực tiếp Player — Boss."""
        if self.boss is None:
            return
        player_rect = pygame.Rect(
            self.player.x, self.player.y,
            self.player.WIDTH, self.player.HEIGHT
        )
        if player_rect.colliderect(self.boss.get_rect()):
            took = self.player.take_damage(now)
            if took:
                self.player_hit_this_step = True
                if self.player.is_dead():
                    self.running = False
                    self.is_victory = False
                    self.end_time = now

    def _update_hit_effects(self):
        now = self._now()
        self.hit_effects = [e for e in self.hit_effects if e[2] > now]

    # ------------------------------------------------------------------
    # Is over
    # ------------------------------------------------------------------

    def is_over(self) -> bool:
        return not self.running

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_survival_time(self) -> str:
        current = self._now() if self.running else (self.end_time or self._now())
        total_s = (current - self.start_time) // 1000
        return f"{total_s // 60:02}:{total_s % 60:02}"

    def set_episode_label(self, label: str):
        self.episode_label = label

    # ------------------------------------------------------------------
    # Draw (render mode)
    # ------------------------------------------------------------------

    def draw(self, screen):
        if self.background:
            screen.blit(self.background, (0, 0))
        else:
            screen.fill(BACKGROUND_COLOR)

        self.player.draw(screen)
        if self.boss:
            self.boss.draw(screen)

        self.boss_bullet_manager.draw(screen)
        self.bullet_manager.draw(screen)
        self.player_bullet_manager.draw(screen)

        for x, y, _ in self.hit_effects:
            if self.hit_effect_sprite:
                screen.blit(self.hit_effect_sprite, (int(x - 24), int(y - 24)))

        self.ui.draw_timer(screen, self.get_survival_time())
        self.ui.draw_health(screen, self.player.health)
        if self.boss:
            self.ui.draw_boss_health(screen, self.boss.health, self.boss.max_health)
        self.ui.draw_episode_label(screen, self.episode_label)
