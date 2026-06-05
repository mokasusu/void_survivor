"""
PPOConfig — cấu hình cho vòng lặp huấn luyện MaskablePPO.
Đọc từ biến môi trường (.env), có giá trị mặc định hợp lý.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)
else:
    load_dotenv()


def _e(name: str, cast, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    if cast is bool:
        return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}
    try:
        return cast(raw)
    except (ValueError, TypeError):
        return default


def _estr(name: str, default=None):
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = str(raw).strip()
    return raw if raw else default


@dataclass
class PPOConfig:

    # --- Game ---
    max_steps: int       = field(default_factory=lambda: _e("PPO_MAX_STEPS", int, 3000))
    render: bool         = field(default_factory=lambda: _e("PPO_RENDER", bool, False))
    render_fps: int      = field(default_factory=lambda: _e("PPO_RENDER_FPS", int, 60))

    # --- Curriculum ---
    total_timesteps: int = field(default_factory=lambda: _e("PPO_TOTAL_TIMESTEPS", int, 5_000_000))
    eval_interval: int   = field(default_factory=lambda: _e("PPO_EVAL_INTERVAL", int, 50_000))
    eval_episodes: int   = field(default_factory=lambda: _e("PPO_EVAL_EPISODES", int, 20))
    promote_threshold: float = field(default_factory=lambda: _e("PPO_PROMOTE_THRESHOLD", float, 0.80))
    old_stage_drop_limit: float = field(default_factory=lambda: _e("PPO_OLD_STAGE_DROP", float, 0.05))
    rollback_threshold: float = field(default_factory=lambda: _e("PPO_ROLLBACK_THRESHOLD", float, 0.15))
    rollback_patience: int = field(default_factory=lambda: _e("PPO_ROLLBACK_PATIENCE", int, 3))

    # --- MaskablePPO hyperparams ---
    n_envs: int          = field(default_factory=lambda: _e("PPO_N_ENVS", int, 8))
    learning_rate: float = field(default_factory=lambda: _e("PPO_LR", float, 3e-4))
    gamma: float         = field(default_factory=lambda: _e("PPO_GAMMA", float, 0.99))
    ent_coef: float      = field(default_factory=lambda: _e("PPO_ENT_COEF", float, 0.05))
    n_steps: int         = field(default_factory=lambda: _e("PPO_N_STEPS", int, 2048))
    batch_size: int      = field(default_factory=lambda: _e("PPO_BATCH_SIZE", int, 256))
    n_epochs: int        = field(default_factory=lambda: _e("PPO_N_EPOCHS", int, 10))
    clip_range: float    = field(default_factory=lambda: _e("PPO_CLIP_RANGE", float, 0.2))

    # --- Dynamic entropy schedule ---
    ent_coef_boost: float = field(default_factory=lambda: _e("PPO_ENT_COEF_BOOST", float, 0.02))
    ent_coef_boost_steps: int = field(default_factory=lambda: _e("PPO_ENT_COEF_BOOST_STEPS", int, 100_000))
    ent_coef_floor: float = field(default_factory=lambda: _e("PPO_ENT_COEF_FLOOR", float, 0.01))

    # --- Paths ---
    models_dir: str      = field(default_factory=lambda: _estr("PPO_MODELS_DIR", "models/ppo"))
    run_name: str | None = field(default_factory=lambda: _estr("PPO_RUN_NAME", None))
    tensorboard_log: str = field(default_factory=lambda: _estr("PPO_TB_LOG", "./tensorboard_logs/"))
