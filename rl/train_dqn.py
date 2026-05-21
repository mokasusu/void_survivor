import argparse
import csv
import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from rl.env import VoidSurvivorEnv
from rl.config import DQNConfig
from rl.replay_buffer import ReplayBuffer
from rl.agent import DQNAgent
from rl.state_encoder import StateEncoder


def _build_run_dir(cfg: DQNConfig) -> Path:
    if cfg.run_name:
        return Path(cfg.models_dir) / cfg.mode / cfg.run_name

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(cfg.models_dir) / cfg.mode / timestamp


def _save_checkpoint(path: Path, agent: DQNAgent, replay: ReplayBuffer, cfg: DQNConfig, episode: int, steps_done: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "episode": episode,
        "steps_done": steps_done,
        "config": cfg.__dict__,
        "policy_state": agent.policy_net.state_dict(),
        "target_state": agent.target_net.state_dict(),
        "optimizer_state": agent.optimizer.state_dict(),
    }
    torch.save(payload, path)


def _load_checkpoint(path: Path, agent: DQNAgent):
    payload = torch.load(path, map_location=agent.device)
    agent.policy_net.load_state_dict(payload["policy_state"])
    agent.target_net.load_state_dict(payload["target_state"])
    agent.optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def _append_metrics_csv(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _save_plots(plot_dir: Path, episodes, rewards, moving_avg, losses, avg_qs, steps, elapsed_seconds):
    plot_dir.mkdir(parents=True, exist_ok=True)

    def _plot_metric(x, y, title, ylabel, filename):
        plt.figure(figsize=(10, 5))
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel("Episode")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(plot_dir / filename)
        plt.close()

    _plot_metric(episodes, rewards, "Episode Reward", "Reward", "reward.png")
    _plot_metric(episodes, moving_avg, "Moving Average Reward", "Reward", "moving_avg_reward.png")
    _plot_metric(episodes, losses, "DQN Loss (avg per episode)", "Loss", "loss.png")
    _plot_metric(episodes, avg_qs, "Average Q-Value (avg per episode)", "Q-Value", "avg_q_value.png")
    _plot_metric(episodes, steps, "Steps per Episode", "Steps", "steps.png")
    _plot_metric(episodes, elapsed_seconds, "Elapsed Seconds per Episode", "Seconds", "elapsed_seconds.png")


def _resolve_resume_path(cfg: DQNConfig) -> Path | None:
    if not cfg.resume_path:
        return None

    path = Path(cfg.resume_path)
    if not path.is_absolute() and not path.exists():
        candidate = Path(cfg.models_dir) / path
        if candidate.exists():
            path = candidate

    return path if path.exists() else None


def train(config=None):

    cfg = config or DQNConfig()
    def _make_env(render_enabled: bool):
        return VoidSurvivorEnv(
            mode=cfg.mode,
            render=render_enabled,
            max_steps=cfg.max_steps,
            render_fps=cfg.render_fps,
            encoder=StateEncoder(max_bullets=cfg.max_bullets)
        )

    def _write_meta(meta_file: Path, episode_num: int, steps: int, best_episode: int, best_score: float):
        with meta_file.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "last_episode": episode_num,
                    "steps_done": steps,
                    "best_episode": best_episode,
                    "best_moving_avg_reward": best_score
                },
                f,
                ensure_ascii=False,
                indent=2
            )

    env = _make_env(cfg.render)
    state_dim = len(env.reset())
    action_dim = len(env.ACTIONS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    agent = DQNAgent(state_dim, action_dim, device, lr=cfg.lr, gamma=cfg.gamma)
    replay = ReplayBuffer(cfg.buffer_size)

    run_dir = _build_run_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt_path = run_dir / "last.pt"
    best_ckpt_path = run_dir / "best.pt"
    meta_path = run_dir / "meta.json"
    metrics_csv_path = run_dir / "metrics.csv"
    plots_dir = run_dir / "plots"

    steps_done = 0
    start_episode = 0
    episode = 0

    moving_avg_window = getattr(cfg, "moving_avg_window", 20)
    reward_window = deque(maxlen=moving_avg_window)
    logged_episodes = []
    logged_rewards = []
    logged_moving_avg = []
    logged_losses = []
    logged_avg_qs = []
    logged_steps = []
    logged_elapsed_seconds = []
    best_moving_avg_reward = float("-inf")
    best_episode = 0

    resume_path = _resolve_resume_path(cfg)
    if resume_path:
        if resume_path.is_dir():
            resume_path = resume_path / "last.pt"
        if resume_path.exists():
            payload = _load_checkpoint(resume_path, agent)
            start_episode = int(payload.get("episode", 0))
            steps_done = int(payload.get("steps_done", 0))

    try:
        for episode in range(start_episode, cfg.episodes):
            render_this_episode = False
            if cfg.render:
                if cfg.render_every and cfg.render_every > 0:
                    render_this_episode = ((episode + 1) % cfg.render_every) == 0
                else:
                    render_this_episode = True

            if render_this_episode != env.render_enabled:
                env.close()
                env = _make_env(render_this_episode)

            env.set_episode_label(
                f"Episode {episode + 1}/{cfg.episodes}"
            )

            state = env.reset()
            episode_reward = 0.0
            episode_loss_sum = 0.0
            episode_q_sum = 0.0
            episode_train_steps = 0
            episode_steps = 0
            action_counts = [0] * action_dim
            last_info = None

            for _ in range(cfg.max_steps):
                epsilon = cfg.epsilon_final + (cfg.epsilon_start - cfg.epsilon_final) * \
                    max(0.0, (cfg.epsilon_decay - steps_done) / cfg.epsilon_decay)

                action = agent.select_action(state, epsilon)
                action_counts[action] += 1
                next_state, reward, done, _ = env.step(action)
                last_info = _
                replay.push(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward
                steps_done += 1
                episode_steps += 1

                if len(replay) >= cfg.batch_size:
                    if cfg.train_every <= 1 or (steps_done % cfg.train_every == 0):
                        batch = replay.sample(cfg.batch_size)
                        loss_value, avg_q_value = agent.train_step(batch)
                        episode_loss_sum += loss_value
                        episode_q_sum += avg_q_value
                        episode_train_steps += 1

                if steps_done % cfg.target_update == 0:
                    agent.sync_target()

                if done:
                    break

            avg_loss = episode_loss_sum / max(1, episode_train_steps)
            avg_q = episode_q_sum / max(1, episode_train_steps)
            reward_window.append(episode_reward)
            moving_avg_reward = sum(reward_window) / max(1, len(reward_window))
            elapsed_seconds = last_info.get("elapsed_seconds", 0) if last_info else 0
            survival_time = last_info.get("survival_time", "00:00") if last_info else "00:00"

            total_actions = max(1, sum(action_counts))
            action_pcts = [count / total_actions for count in action_counts]
            action_dist_str = " ".join(
                f"{idx}:{pct:.0%}"
                for idx, pct in enumerate(action_pcts)
            )

            print(
                "Episode {}/{} | Reward: {:.2f} | MA Reward: {:.2f} | Loss: {:.4f} | Avg Q: {:.4f} | Steps: {} | Survival: {} | Actions: {}".format(
                    episode + 1,
                    cfg.episodes,
                    episode_reward,
                    moving_avg_reward,
                    avg_loss,
                    avg_q,
                    episode_steps,
                    survival_time,
                    action_dist_str
                )
            )

            metrics_row = {
                "episode": episode + 1,
                "reward": episode_reward,
                "moving_avg_reward": moving_avg_reward,
                "loss": avg_loss,
                "avg_q_value": avg_q,
                "steps": episode_steps,
                "elapsed_seconds": elapsed_seconds,
                "survival_time": survival_time,
            }
            for idx, count in enumerate(action_counts):
                metrics_row[f"action_{idx}_count"] = count
                metrics_row[f"action_{idx}_pct"] = action_pcts[idx]

            _append_metrics_csv(metrics_csv_path, metrics_row)

            logged_episodes.append(episode + 1)
            logged_rewards.append(episode_reward)
            logged_moving_avg.append(moving_avg_reward)
            logged_losses.append(avg_loss)
            logged_avg_qs.append(avg_q)
            logged_steps.append(episode_steps)
            logged_elapsed_seconds.append(elapsed_seconds)

            if moving_avg_reward > best_moving_avg_reward:
                best_moving_avg_reward = moving_avg_reward
                best_episode = episode + 1
                _save_checkpoint(best_ckpt_path, agent, replay, cfg, episode + 1, steps_done)

            if (episode + 1) % cfg.save_every_episodes == 0:
                _save_checkpoint(last_ckpt_path, agent, replay, cfg, episode + 1, steps_done)
                _write_meta(meta_path, episode + 1, steps_done, best_episode, best_moving_avg_reward)

            if cfg.episode_delay > 0:
                time.sleep(cfg.episode_delay)
    except KeyboardInterrupt:
        _save_checkpoint(last_ckpt_path, agent, replay, cfg, episode + 1, steps_done)
        _write_meta(meta_path, episode + 1, steps_done, best_episode, best_moving_avg_reward)
        _save_plots(plots_dir, logged_episodes, logged_rewards, logged_moving_avg, logged_losses, logged_avg_qs, logged_steps, logged_elapsed_seconds)
        print("Interrupted. Checkpoint saved.")
    finally:
        _save_checkpoint(last_ckpt_path, agent, replay, cfg, episode + 1, steps_done)
        _write_meta(meta_path, episode + 1, steps_done, best_episode, best_moving_avg_reward)
        _save_plots(plots_dir, logged_episodes, logged_rewards, logged_moving_avg, logged_losses, logged_avg_qs, logged_steps, logged_elapsed_seconds)
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default=None, help="Path to checkpoint or run directory")
    args = parser.parse_args()

    cfg = DQNConfig()
    if args.checkpoint:
        cfg.resume_path = args.checkpoint
    train(cfg)
