"""
Boss stage configuration cho Curriculum Learning.

Stage 1: Boss đứng yên, HP thấp, không bắn.
Stage 2: Boss đứng yên, bắn đạn thẳng, tần suất chậm.
Stage 3: Boss di chuyển qua lại, tần suất bắn tăng, đạn nhanh hơn.
Stage 4: Boss di chuyển thông minh, bắn 2-3 viên độc lập (không đuổi).
Stage 5: Boss di chuyển thông minh, bắn đạn chùm lớn (spread_large) hướng thẳng về phía Agent.
Stage 6: Boss di chuyển thông minh, kết hợp Stage 4 và 5 qua cơ chế "Lệch pha tuần hoàn" (timers độc lập).
"""

import math

from config.settings import WIDTH, HEIGHT, INFO_PANEL_HEIGHT


# ---------------------------------------------------------------------------
# Stage config constants
# ---------------------------------------------------------------------------

STAGE_CONFIGS = {
    1: {
        "hp": 100,
        "moves": False,
        "move_speed": 0,
        "shoot": False,
        "shoot_interval_frames": 0,   # không bắn
        "bullet_speed": 0.0,
        "pattern": "none",
    },
    2: {
        "hp": 150,
        "moves": False,
        "move_speed": 0,
        "shoot": True,
        "shoot_interval_frames": 120,  # 2 giây tại 60fps
        "bullet_speed": 5.0,
        "pattern": "straight",
    },
    3: {
        "hp": 200,
        "moves": True,
        "move_speed": 2.0,
        "shoot": True,
        "shoot_interval_frames": 60,   # 1 giây tại 60fps
        "bullet_speed": 7.0,
        "pattern": "straight",
    },
    4: {
        "hp": 250,
        "moves": True,
        "move_speed": 3.0,
        "shoot": True,
        "shoot_interval_frames": 80,   # 1.33 giây tại 60fps
        "bullet_speed": 7.5,
        "pattern": "independent",       # bắn 2-3 viên độc lập (không đuổi)
    },
    5: {
        "hp": 300,
        "moves": True,
        "move_speed": 3.5,
        "shoot": True,
        "shoot_interval_frames": 120,  # đạn chùm bắn về phía agent (2s)
        "bullet_speed": 7.5,
        "pattern": "spread_large",
    },
    6: {
        "hp": 300,
        "moves": True,
        "move_speed": 4.0,
        "shoot": True,
        "shoot_interval_frames": 120,  # Kết hợp Stage 4 & 5 qua timers độc lập
        "bullet_speed": 8.0,
        "pattern": "periodic_stage6",
    },
}


def get_stage_config(stage: int) -> dict:
    """Trả về config cho stage, clamp về [1, 6]."""
    stage = max(1, min(6, stage))
    return STAGE_CONFIGS[stage]


# ---------------------------------------------------------------------------
# Bullet spawning helpers (trả về list[dict] — mỗi dict là 1 viên đạn)
# Mỗi viên: {"x": float, "y": float, "vx": float, "vy": float}
# ---------------------------------------------------------------------------

def _spawn_straight(boss_cx: float, boss_cy: float, target_cx: float, target_cy: float, speed: float) -> list:
    """Bắn thẳng hướng về phía Agent."""
    dx = target_cx - boss_cx
    dy = target_cy - boss_cy
    dist = math.hypot(dx, dy) or 1.0
    return [{"x": boss_cx, "y": boss_cy, "vx": dx / dist * speed, "vy": dy / dist * speed}]


def _spawn_spread_light(boss_cx: float, boss_cy: float, target_cx: float, target_cy: float, speed: float) -> list:
    """Bắn chùm nhẹ: 3 viên với góc rộng hơn (±20°)."""
    dx = target_cx - boss_cx
    dy = target_cy - boss_cy
    base_angle = math.atan2(dy, dx)
    bullets = []
    for delta_deg in [-20, 0, 20]:
        angle = base_angle + math.radians(delta_deg)
        bullets.append({
            "x": boss_cx,
            "y": boss_cy,
            "vx": math.cos(angle) * speed,
            "vy": math.sin(angle) * speed,
        })
    return bullets


def _spawn_spread_large(boss_cx: float, boss_cy: float, target_cx: float, target_cy: float, speed: float) -> list:
    """Bắn chùm 5-6 viên: bắn 5 viên trải đều từ -30° đến 30°."""
    dx = target_cx - boss_cx
    dy = target_cy - boss_cy
    base_angle = math.atan2(dy, dx)
    bullets = []
    for delta_deg in [-30, -15, 0, 15, 30]:
        angle = base_angle + math.radians(delta_deg)
        bullets.append({
            "x": boss_cx,
            "y": boss_cy,
            "vx": math.cos(angle) * speed,
            "vy": math.sin(angle) * speed,
        })
    return bullets


def spawn_boss_bullets(pattern: str, boss_cx: float, boss_cy: float,
                       target_cx: float, target_cy: float, speed: float) -> list:
    """
    Sinh đạn dựa trên pattern của stage.
    Trả về list các dict mô tả viên đạn.
    """
    if pattern == "none" or speed <= 0:
        return []
    if pattern == "straight":
        return _spawn_straight(boss_cx, boss_cy, target_cx, target_cy, speed)
    if pattern == "independent":
        return _spawn_straight(boss_cx, boss_cy, target_cx, target_cy, speed)
    if pattern == "spread_light":
        return _spawn_spread_light(boss_cx, boss_cy, target_cx, target_cy, speed)
    if pattern == "spread_large":
        return _spawn_spread_large(boss_cx, boss_cy, target_cx, target_cy, speed)
    return _spawn_straight(boss_cx, boss_cy, target_cx, target_cy, speed)
