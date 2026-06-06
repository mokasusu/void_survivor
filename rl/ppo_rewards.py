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
        
        # GIẢI PHÁP ĐẶC TRỊ: Biến đếm thời gian không gây sát thương cho Boss
        self.frames_since_last_damage: int = 0

    def reset(self, game):
        self._prev_hp = game.player.health
        self._prev_boss_hp = game.boss.health if game.boss else 0
        self._prev_player_bullets = len(game.player_bullet_manager.bullets)
        self.frames_since_last_damage = 0

    def compute(self, game, stage: int = 1, stage_steps: int = 0) -> tuple[float, dict]:
        breakdown = {
            "frame": 0.0,
            "hit_penalty": 0.0,
            "safe_distance": 0.0,
            "boss_damage": 0.0,
            "miss_penalty": 0.0,
            "alignment": 0.0,
            "stagnation_penalty": 0.0,  # Luồng phạt câu giờ
            "terminal": 0.0,
        }

        # ---------------------------------------------------------------
        # R_survival — Frame reward (Hạ thấp xuống để tránh bào điểm câu giờ)
        # ---------------------------------------------------------------
        if game.running:
            breakdown["frame"] += 0.002  # Giảm từ 0.01 xuống 0.002

        # ---------------------------------------------------------------
        # R_survival — Phạt trúng đạn (Bảo hiểm y tế: giảm 50% đầu Stage 2)
        # ---------------------------------------------------------------
        hp_loss = self._prev_hp - game.player.health
        if hp_loss > 0:
            penalty_coef = 2.0
            if stage == 2:
                # 12,500 steps/env tương ứng 100,000 global steps
                p = min(1.0, max(0.0, stage_steps / 12500))
                # Phạt trúng đạn bắt đầu từ 1.0 (giảm 50%) tăng dần về 2.0
                penalty_coef *= (0.5 + 0.5 * p)
            breakdown["hit_penalty"] -= penalty_coef * hp_loss

        # ---------------------------------------------------------------
        # R_survival — Phạt khoảng cách gần đạn (Chỉ phạt khi đạn lọt vào vùng nguy hiểm)
        # ---------------------------------------------------------------
        agent_cx = game.player.x + game.player.WIDTH / 2
        agent_cy = game.player.y + game.player.HEIGHT / 2
        closest_dist = self._closest_bullet_dist(game, agent_cx, agent_cy)
        proximity_radius = 60.0
        if 0 < closest_dist < proximity_radius:
            # Đạn càng sát người phạt càng nặng (tối đa -0.05)
            breakdown["safe_distance"] -= 0.05 * (1.0 - (closest_dist / proximity_radius))

        # ---------------------------------------------------------------
        # R_alignment — Thưởng đứng thẳng hàng theo trục Y (Alignment Weight Boost đầu Stage 3)
        # ---------------------------------------------------------------
        if game.boss and game.running:
            boss_cy = game.boss.y + game.boss.display_height / 2
            align_coef = 0.003
            if stage == 3:
                # 12,500 steps/env tương ứng 100,000 global steps
                p = min(1.0, max(0.0, stage_steps / 12500))
                # Tăng gấp đôi alignment reward lúc bắt đầu stage 3 (0.006) rồi giảm dần về mặc định (0.003)
                align_coef *= (2.0 - 1.0 * p)
            breakdown["alignment"] += align_coef * max(0.0, 1.0 - abs(agent_cy - boss_cy) / HEIGHT)

        # ---------------------------------------------------------------
        # R_offensive — Thưởng sát thương Boss & Reset bộ đếm câu giờ
        # ---------------------------------------------------------------
        boss_hp_loss = self._prev_boss_hp - (game.boss.health if game.boss else 0)
        if boss_hp_loss > 0:
            breakdown["boss_damage"] += 0.6 * boss_hp_loss  # Cân bằng lại theo cooldown 6 (1.5 * 6 / 16 = 0.56)
            self.frames_since_last_damage = 0  # Gây damage thành công -> Reset bộ đếm trì trệ
        else:
            if game.running:
                self.frames_since_last_damage += 1

        # ---------------------------------------------------------------
        # GIẢI PHÁP ĐẶC TRỊ: Phạt câu giờ / Trì trệ (Stagnation Penalty)
        # ---------------------------------------------------------------
        # Nếu quá 3 giây (180 frames ở 60fps) không bắn trúng Boss, bắt đầu phạt nặng
        if self.frames_since_last_damage > 180:
            breakdown["stagnation_penalty"] -= 0.02
            # Nếu quá 8 giây (480 frames) cực hình, phạt lũy tiến tăng cường
            if self.frames_since_last_damage > 480:
                breakdown["stagnation_penalty"] -= 0.1

        # ---------------------------------------------------------------
        # R_offensive — Phạt bắn hụt (Đã sửa logic tách biệt hoàn toàn)
        # ---------------------------------------------------------------
        cur_player_bullets = len(game.player_bullet_manager.bullets)
        # Số lượng đạn thực tế đã biến mất khỏi bộ quản lý trong frame này
        bullets_destroyed = self._prev_player_bullets - cur_player_bullets
        
        if bullets_destroyed > 0:
            # Số đạn hụt = Tổng số đạn biến mất trừ đi số đạn thực sự trúng vào người Boss
            # (Giả định mỗi viên đạn trúng gây 1 damage. Nếu đạn gây n damage, hãy chia cho n)
            actual_missed = max(0, bullets_destroyed - boss_hp_loss)
            if actual_missed > 0:
                breakdown["miss_penalty"] -= 0.02 * actual_missed

        # ---------------------------------------------------------------
        # R_terminal
        # ---------------------------------------------------------------
        if not game.running:
            if game.is_victory:
                breakdown["terminal"] += 100.0
            else:
                # Nếu chết do hết HP hoặc bị timeout hệ thống ép ngắt
                breakdown["terminal"] -= 20.0

        # ---------------------------------------------------------------
        # Cập nhật trạng thái cho frame kế tiếp
        # ---------------------------------------------------------------
        self._prev_hp = game.player.health
        self._prev_boss_hp = game.boss.health if game.boss else 0
        self._prev_player_bullets = cur_player_bullets

        total = sum(breakdown.values())
        return total, breakdown

    def _closest_bullet_dist(self, game, agent_cx: float, agent_cy: float) -> float:
        all_bullets = []
        if hasattr(game, "boss_bullet_manager"):
            all_bullets.extend(game.boss_bullet_manager.bullets)

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
