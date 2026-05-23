import argparse
from pathlib import Path

import torch

from rl.env import VoidSurvivorEnv
from rl.config import DQNConfig
from rl.agent import DQNAgent
from rl.state_encoder import StateEncoder


REWARD_KEYS = [
    "survival",
    "time_pressure",
    "danger",
    "damage_penalty",
    "boss_damage",
    "win_loss",
]


def _format_reward_breakdown(breakdown: dict):
    header = " | ".join(REWARD_KEYS)
    values = " | ".join(
        f"{breakdown.get(key, 0.0):.3f}" for key in REWARD_KEYS
    )
    return header, values


def _resolve_checkpoint(path_value: str | None, models_dir: str):
    if not path_value:
        return None

    path = Path(path_value)
    if not path.is_absolute() and not path.exists():
        candidate = Path(models_dir) / path
        if candidate.exists():
            path = candidate

    if path.is_dir():
        best_path = path / "best.pt"
        last_path = path / "last.pt"
        if best_path.exists():
            return best_path
        if last_path.exists():
            return last_path
        return None

    return path if path.exists() else None


def _load_checkpoint(path: Path, agent: DQNAgent):
    payload = torch.load(path, map_location=agent.device)
    agent.policy_net.load_state_dict(payload["policy_state"])
    agent.target_net.load_state_dict(payload["target_state"])
    agent.optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def test(config=None, checkpoint_path: str | None = None, episodes: int = 5, render: bool = True):
    cfg = config or DQNConfig()

    ckpt_path = _resolve_checkpoint(checkpoint_path or cfg.resume_path, cfg.models_dir)
    if ckpt_path is None:
        raise FileNotFoundError("Checkpoint not found. Provide a valid file or directory path.")

    def _make_env(render_enabled: bool):
        return VoidSurvivorEnv(
            mode=cfg.mode,
            difficulty=cfg.difficulty,
            render=render_enabled,
            max_steps=cfg.max_steps,
            render_fps=cfg.render_fps,
            encoder=StateEncoder(max_bullets=cfg.max_bullets),
            auto_fire=cfg.auto_fire
        )

    env = _make_env(render)
    state_dim = len(env.reset())
    action_dim = len(env.actions)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(state_dim, action_dim, device, lr=cfg.lr, gamma=cfg.gamma)
    _load_checkpoint(ckpt_path, agent)

    total_rewards = []
    total_steps = []

    for episode in range(episodes):
        env.set_episode_label(
            f"Test {episode + 1}/{episodes}"
        )
        state = env.reset()
        episode_reward = 0.0
        steps = 0
        reward_breakdown_sum = {key: 0.0 for key in REWARD_KEYS}
        last_info = None

        for _ in range(cfg.max_steps):
            action = agent.select_action(state, 0.0)
            next_state, reward, done, info = env.step(action)
            state = next_state
            episode_reward += reward
            step_breakdown = info.get("reward_breakdown", {}) if info else {}
            for key in REWARD_KEYS:
                reward_breakdown_sum[key] += step_breakdown.get(key, 0.0)
            steps += 1
            last_info = info
            if done:
                break

        survival_time = last_info.get("survival_time", "00:00") if last_info else "00:00"
        print(
            "Test Episode {}/{}  Reward: {:.2f}  Steps: {}  Survival: {}".format(
                episode + 1,
                episodes,
                episode_reward,
                steps,
                survival_time
            )
        )
        if episode == 0:
            header, _ = _format_reward_breakdown(reward_breakdown_sum)
            print(header)
        _, values = _format_reward_breakdown(reward_breakdown_sum)
        print(values)
        print("")
        total_rewards.append(episode_reward)
        total_steps.append(steps)

    avg_reward = sum(total_rewards) / max(1, len(total_rewards))
    avg_steps = sum(total_steps) / max(1, len(total_steps))
    print(
        "Test Summary | Avg Reward: {:.2f} | Avg Steps: {:.0f}".format(
            avg_reward,
            avg_steps
        )
    )
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default=None, help="Path to checkpoint or run directory")
    args = parser.parse_args()

    if args.checkpoint:
        test(checkpoint_path=args.checkpoint)
    else:
        test()
