import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)
else:
    load_dotenv()


def _get_env(name, cast, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    if cast is bool:
        return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}
    if cast is int:
        try:
            return int(raw)
        except ValueError:
            return default
    if cast is float:
        try:
            return float(raw)
        except ValueError:
            return default
    if cast is str:
        return raw
    return default


def _get_env_optional_str(name, default=None):
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = str(raw).strip()
    return raw if raw else default


@dataclass
class DQNConfig:

    mode: str = _get_env("DQN_MODE", str, "boss")
    episodes: int = _get_env("DQN_EPISODES", int, 1000)
    gamma: float = _get_env("DQN_GAMMA", float, 0.99)
    lr: float = _get_env("DQN_LR", float, 1e-3)
    batch_size: int = _get_env("DQN_BATCH_SIZE", int, 64)
    buffer_size: int = _get_env("DQN_BUFFER_SIZE", int, 50000)
    target_update: int = _get_env("DQN_TARGET_UPDATE", int, 500)
    epsilon_start: float = _get_env("DQN_EPSILON_START", float, 0.1)
    epsilon_final: float = _get_env("DQN_EPSILON_FINAL", float, 0.05)
    epsilon_decay: int = _get_env("DQN_EPSILON_DECAY", int, 20000)
    max_steps: int = _get_env("DQN_MAX_STEPS", int, 5000)
    render: bool = _get_env("DQN_RENDER", bool, True)
    render_every: int = _get_env("DQN_RENDER_EVERY", int, 0)
    episode_delay: float = _get_env("DQN_EPISODE_DELAY", float, 0)
    render_fps: int = _get_env("DQN_RENDER_FPS", int, 60)
    models_dir: str = _get_env("DQN_MODELS_DIR", str, "models")
    run_name: str | None = _get_env_optional_str("DQN_RUN_NAME", None)
    resume_path: str | None = _get_env_optional_str("DQN_RESUME_PATH", None)
    save_every_episodes: int = _get_env("DQN_SAVE_EVERY_EPISODES", int, 1)
    moving_avg_window: int = _get_env("DQN_MOVING_AVG_WINDOW", int, 20)
    max_bullets: int = _get_env("DQN_MAX_BULLETS", int, 300)
    train_every: int = _get_env("DQN_TRAIN_EVERY", int, 4)
    grad_clip: float | None = _get_env("DQN_GRAD_CLIP", float, 5.0)
