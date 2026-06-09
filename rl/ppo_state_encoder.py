"""
PPOStateEncoder — mã hóa trạng thái game thành vector 45 chiều theo spec void_survivor.md.

Cấu trúc:
  [0..1]   Agent Status: [HP_norm, Cooldown_norm]
  [2..4]   Boss Status:  [dx/W, dy/H, boss_hp_norm]
  [5..44]  Bullet Radar: 10 viên gần nhất x [(dx/W, dy/H, vx/Vmax, vy/Vmax)]

Tất cả tọa độ là TƯƠNG ĐỐI so với vị trí Agent, chuẩn hóa về [-1, 1] hoặc [0, 1].
"""

import math
import numpy as np

from config.settings import WIDTH, HEIGHT

# Tốc độ đạn tối đa trong cấu hình game (Stage 4 spread ~8 px/frame)
V_MAX = 12.0
D_MAX = math.sqrt(WIDTH ** 2 + HEIGHT ** 2)

MAX_BULLETS = 10  # Bullet Radar = 10 viên gần nhất
SHOOT_COOLDOWN_MAX = 6  # frames, khớp với game.py


class PPOStateEncoder:
    """
    Encode game state → np.float32 array shape (45,).
    Nhận `game_ppo` là instance của GamePPO.
    """

    STATE_DIM = 45

    def encode(self, game) -> np.ndarray:
        obs = np.zeros(self.STATE_DIM, dtype=np.float32)

        agent = game.player
        agent_cx = agent.x + agent.WIDTH / 2
        agent_cy = agent.y + agent.HEIGHT / 2

        # --- [0..1] Agent Status ---
        obs[0] = agent.health / max(1, agent.max_health)
        obs[1] = game.shoot_cooldown / SHOOT_COOLDOWN_MAX

        # --- [2..4] Boss Status ---
        boss = game.boss
        if boss is not None:
            boss_cx, boss_cy = boss.get_center()
            obs[2] = (boss_cx - agent_cx) / max(1, WIDTH)
            obs[3] = (boss_cy - agent_cy) / max(1, HEIGHT)
            obs[4] = boss.health / max(1, boss.max_health)

        # --- [5..44] Bullet Radar (10 viên gần nhất) ---
        bullets = game.boss_bullet_manager.bullets if hasattr(game, "boss_bullet_manager") else []

        # Thêm đạn từ survival bullet_manager nếu có
        env_bullets = getattr(game.bullet_manager, "bullets", [])

        # Gộp tất cả đạn và chọn 10 viên gần nhất
        all_bullets = list(bullets) + list(env_bullets)

        def _dist_sq(b):
            bx = getattr(b, "x", 0)
            by = getattr(b, "y", 0)
            return (bx - agent_cx) ** 2 + (by - agent_cy) ** 2

        closest = sorted(all_bullets, key=_dist_sq)[:MAX_BULLETS]

        for i, b in enumerate(closest):
            base = 5 + i * 4
            bx = getattr(b, "x", 0)
            by = getattr(b, "y", 0)
            # velocity
            vx = getattr(b, "vx", getattr(b, "speed", 0.0))
            vy = getattr(b, "vy", 0.0)
            obs[base + 0] = (bx - agent_cx) / max(1, WIDTH)
            obs[base + 1] = (by - agent_cy) / max(1, HEIGHT)
            obs[base + 2] = vx / V_MAX
            obs[base + 3] = vy / V_MAX

        return obs
