"""
PPORewardShaper — hàm Reward theo spec void_survivor.md (Mục I.3).

Công thức tổng quát:
    Reward = R_survival + R_offensive + R_terminal

R_survival:
    +0.005  mỗi frame sống sót
    -1.0    khi bị trúng đạn (HP giảm)
    +0.01 * (dist_closest_bullet / D_max)  thưởng khoảng cách an toàn

R_offensive:
    +0.2 * damage_dealt   thưởng sát thương lên Boss
    -0.05                 phạt bắn hụt (đạn ra ngoài mà không trúng Boss)

R_terminal:
    +50.0   Boss chết
    -20.0   Agent chết
"""

import math

from config.settings import WIDTH, HEIGHT

D_MAX = math.sqrt(WIDTH ** 2 + HEIGHT ** 2)


class PPORewardShaper:

    def __init__(self):
        self._prev_hp: int = 3
        self._prev_boss_hp: int = 0
        self._prev_player_bullets: int = 0

    # ------------------------------------------------------------------
    # Reset tại đầu mỗi episode
    # ------------------------------------------------------------------

    def reset(self, game):
        self._prev_hp = game.player.health
        self._prev_boss_hp = game.boss.health if game.boss else 0
        self._prev_player_bullets = len(game.player_bullet_manager.bullets)

    # ------------------------------------------------------------------
    # Compute reward tại mỗi step
    # ------------------------------------------------------------------

    def compute(self, game) -> tuple[float, dict]:
        breakdown = {
            "frame": 0.0,
            "hit_penalty": 0.0,
            "safe_distance": 0.0,
            "boss_damage": 0.0,
            "miss_penalty": 0.0,
            "terminal": 0.0,
        }

        # ---------------------------------------------------------------
        # R_survival — Frame reward
        # ---------------------------------------------------------------
        if game.running:
            breakdown["frame"] += 0.005

        # ---------------------------------------------------------------
        # R_survival — Phạt trúng đạn
        # ---------------------------------------------------------------
        hp_loss = self._prev_hp - game.player.health
        if hp_loss > 0:
            breakdown["hit_penalty"] -= 1.0 * hp_loss

        # ---------------------------------------------------------------
        # R_survival — Thưởng khoảng cách an toàn đến viên đạn gần nhất
        # ---------------------------------------------------------------
        agent_cx = game.player.x + game.player.WIDTH / 2
        agent_cy = game.player.y + game.player.HEIGHT / 2
        closest_dist = self._closest_bullet_dist(game, agent_cx, agent_cy)
        if closest_dist > 0:
            breakdown["safe_distance"] += 0.01 * (closest_dist / D_MAX)

        # ---------------------------------------------------------------
        # R_offensive — Thưởng sát thương Boss
        # ---------------------------------------------------------------
        boss_hp_loss = self._prev_boss_hp - (game.boss.health if game.boss else 0)
        if boss_hp_loss > 0:
            breakdown["boss_damage"] += 0.2 * boss_hp_loss

        # ---------------------------------------------------------------
        # R_offensive — Phạt bắn hụt
        # Phát hiện: số player bullet giảm (đạn mất đi) mà Boss HP không giảm
        # ---------------------------------------------------------------
        cur_player_bullets = len(game.player_bullet_manager.bullets)
        bullets_fired_and_missed = (
            self._prev_player_bullets - cur_player_bullets - boss_hp_loss
        )
        # bullets_fired_and_missed > 0 nghĩa là có đạn bay mất mà không trúng Boss
        if bullets_fired_and_missed > 0 and boss_hp_loss == 0:
            breakdown["miss_penalty"] -= 0.05 * bullets_fired_and_missed

        # ---------------------------------------------------------------
        # R_terminal
        # ---------------------------------------------------------------
        if not game.running:
            if game.is_victory:
                breakdown["terminal"] += 50.0
            else:
                breakdown["terminal"] -= 20.0

        # ---------------------------------------------------------------
        # Cập nhật prev state
        # ---------------------------------------------------------------
        self._prev_hp = game.player.health
        self._prev_boss_hp = game.boss.health if game.boss else 0
        self._prev_player_bullets = cur_player_bullets

        total = sum(breakdown.values())
        return total, breakdown

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _closest_bullet_dist(self, game, agent_cx: float, agent_cy: float) -> float:
        all_bullets = []
        if hasattr(game, "boss_bullet_manager"):
            all_bullets.extend(game.boss_bullet_manager.bullets)
        all_bullets.extend(getattr(game.bullet_manager, "bullets", []))

        if not all_bullets:
            return 0.0

        min_dist = math.inf
        for b in all_bullets:
            bx = getattr(b, "x", 0)
            by = getattr(b, "y", 0)
            d = math.hypot(bx - agent_cx, by - agent_cy)
            if d < min_dist:
                min_dist = d

        return min_dist if math.isfinite(min_dist) else 0.0
