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
        self._prev_hp: int = 1
        self._prev_boss_hp: int = 0
        self._prev_player_bullets: int = 0
        self._prev_stage: int = 1  # Theo dõi để reset bộ đếm khi qua màn
        self.frames_since_last_damage: int = 0

    def reset(self, game):
        self._prev_hp = game.player.health
        self._prev_boss_hp = game.boss.health if game.boss else 0
        self._prev_player_bullets = len(game.player_bullet_manager.bullets)
        self._prev_stage = 1
        self.frames_since_last_damage = 0

    def compute(self, game, stage: int = 1, stage_steps: int = 0, reward_scale: float = 1.0) -> tuple[float, dict]:
        breakdown = {
            "frame": 0.0,
            "hit_penalty": 0.0,
            "safe_distance": 0.0,
            "grazing": 0.0,
            "boss_damage": 0.0,
            "miss_penalty": 0.0,
            "alignment": 0.0,
            "stagnation_penalty": 0.0, 
            "time_penalty": 0.0,        
            "terminal": 0.0,
        }

        # ---------------------------------------------------------------
        # PHÒNG VỆ CHUYỂN STAGE: Nếu đổi Stage, reset ngay bộ đếm để tránh lỗi tràn số
        # ---------------------------------------------------------------
        if stage != self._prev_stage:
            self._prev_stage = stage
            self._prev_boss_hp = game.boss.health if game.boss else 0
            self._prev_player_bullets = len(game.player_bullet_manager.bullets)
            self.frames_since_last_damage = 0
            return 0.0, breakdown  # Bỏ qua frame bản lề để an toàn toán học

        # Phòng vệ khởi tạo HP: Nếu phát hiện HP hiện tại lớn hơn HP cũ (đầu trận hoặc hồi máu)
        if game.player.health > self._prev_hp:
            self._prev_hp = game.player.health

        # R_survival — Sống sót cơ bản
        if game.running:
            breakdown["frame"] += 0.002  

        # R_survival — Phạt trúng đạn
        hp_loss = self._prev_hp - game.player.health
        if hp_loss > 0:
            penalty_coef = 2.0
            if stage == 2:
                p = min(1.0, max(0.0, stage_steps / 12500))
                penalty_coef *= (0.5 + 0.5 * p)
            breakdown["hit_penalty"] -= penalty_coef * hp_loss

        # R_survival — Phạt khoảng cách gần đạn
        agent_cx = game.player.x + game.player.WIDTH / 2
        agent_cy = game.player.y + game.player.HEIGHT / 2
        closest_dist = self._closest_bullet_dist(game, agent_cx, agent_cy)
        proximity_radius = 60.0
        if 0 < closest_dist < proximity_radius:
            breakdown["safe_distance"] -= 0.05 * (1.0 - (closest_dist / proximity_radius))

        # Thưởng suýt chết (Grazing Reward)
        # Hitbox thực xấp xỉ khoảng cách từ tâm đến cạnh là 15px.
        # Vùng Grazing bọc quanh với bán kính 50px từ tâm.
        grazing_reward = 0.0
        if 15.0 < closest_dist <= 50.0:
            grazing_reward = 0.05
        breakdown["grazing"] = grazing_reward

        # R_alignment — Thưởng đứng thẳng hàng Boss
        if game.boss and game.running:
            boss_cy = game.boss.y + game.boss.display_height / 2
            align_coef = 0.003
            if stage == 3:
                p = min(1.0, max(0.0, stage_steps / 12500))
                align_coef *= (2.0 - 1.0 * p)
            breakdown["alignment"] += align_coef * max(0.0, 1.0 - abs(agent_cy - boss_cy) / HEIGHT)

        # R_offensive — Thưởng sát thương Boss (Đã ép sàn max với 0)
        raw_boss_hp_loss = self._prev_boss_hp - (game.boss.health if game.boss else 0)
        boss_hp_loss = max(0, raw_boss_hp_loss)  # Triệt tiêu hoàn toàn số âm khi Boss đổi stage hồi máu
        
        if boss_hp_loss > 0:
            # Potential-based Reward Scaling: nhân boss_damage với reward_scale
            # Stage 1 = ×1.0, Stage 6 = ×2.25 — giải quyết Sparse Reward ở Stage cao
            breakdown["boss_damage"] += 0.6 * boss_hp_loss * reward_scale
            self.frames_since_last_damage = 0 
        else:
            if game.running:
                self.frames_since_last_damage += 1

        # ---------------------------------------------------------------
        # GIẢI PHÁP MƯỢT MÀ: Phạt trì trệ dạng tuyến tính (Linear Ramp), không dùng hàm bậc thang
        # ---------------------------------------------------------------
        stagnation_grace = 180 + (stage - 1) * 120 
        if self.frames_since_last_damage > stagnation_grace:
            overshoot = self.frames_since_last_damage - stagnation_grace
            # Phạt tăng dần cực kỳ mịn màng dựa trên số frame trễ, tối đa chỉ -0.02 mỗi frame
            breakdown["stagnation_penalty"] -= min(0.02, overshoot * 0.0001)

        # Time Penalty: Phạt thời gian để triệt tiêu hành vi đứng im né đạn ở góc (-0.015/frame)
        if game.running:
            breakdown["time_penalty"] -= 0.015

        # R_offensive — Phạt bắn hụt (Đã vá lỗi logic và giới hạn trần phạt)
        cur_player_bullets = len(game.player_bullet_manager.bullets)
        bullets_destroyed = self._prev_player_bullets - cur_player_bullets
        
        if bullets_destroyed > 0:
            actual_missed = max(0, bullets_destroyed - boss_hp_loss)
            if actual_missed > 0:
                # Giới hạn mức phạt tối đa mỗi frame (-0.1) để tránh việc dọn đạn hàng loạt làm sập mạng Value
                breakdown["miss_penalty"] -= min(0.1, 0.02 * actual_missed)

        # R_terminal (Phạt nhẹ khi chết để định hướng, tránh sụp đổ hàm Value)
        if not game.running:
            if game.is_victory:
                breakdown["terminal"] += 100.0
            else:
                breakdown["terminal"] -= 2.0

        # Cập nhật trạng thái
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