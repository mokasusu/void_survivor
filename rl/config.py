from dataclasses import dataclass


@dataclass
class DQNConfig:

    mode: str = "survival"
    episodes: int = 300
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 50000
    target_update: int = 500
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay: int = 20000
    max_steps: int = 5000
    render: bool = False
