import torch

from rl.env import VoidSurvivorEnv
from rl.config import DQNConfig
from rl.replay_buffer import ReplayBuffer
from rl.agent import DQNAgent


def train(config=None):

    cfg = config or DQNConfig()
    env = VoidSurvivorEnv(mode=cfg.mode, render=cfg.render, max_steps=cfg.max_steps)
    state_dim = len(env.reset())
    action_dim = len(env.ACTIONS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(state_dim, action_dim, device, lr=cfg.lr, gamma=cfg.gamma)
    replay = ReplayBuffer(cfg.buffer_size)

    steps_done = 0

    for episode in range(cfg.episodes):
        state = env.reset()
        episode_reward = 0.0

        for _ in range(cfg.max_steps):
            epsilon = cfg.epsilon_final + (cfg.epsilon_start - cfg.epsilon_final) * \
                max(0.0, (cfg.epsilon_decay - steps_done) / cfg.epsilon_decay)

            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            replay.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            steps_done += 1

            if len(replay) >= cfg.batch_size:
                batch = replay.sample(cfg.batch_size)
                agent.train_step(batch)

            if steps_done % cfg.target_update == 0:
                agent.sync_target()

            if done:
                break

        print(f"Episode {episode + 1}/{cfg.episodes} | Reward: {episode_reward:.2f} | Epsilon: {epsilon:.3f}")

    torch.save(agent.policy_net.state_dict(), f"dqn_{cfg.mode}.pth")
    env.close()


if __name__ == "__main__":
    train()
