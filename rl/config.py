from dataclasses import dataclass


@dataclass
class DQNConfig:

    mode: str = "boss"
    episodes: int = 2000
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 50000
    target_update: int = 500
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay: int = 20000
    max_steps: int = 5000
    render: bool = True
    episode_delay: float = 0.5
    render_fps: int = 360
    models_dir: str = "models"
    run_name: str | None = None
    resume_path: str | None = None
    save_every_episodes: int = 1
    moving_avg_window: int = 20
