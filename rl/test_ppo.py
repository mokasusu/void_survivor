"""Quick smoke-test: import và chạy thử BulletHellEnv."""
import sys
from pathlib import Path

# Project root = thư mục cha của rl/
_ROOT = str(Path(__file__).resolve().parents[1])
# rl/ dir tự động được Python thêm vào sys.path[0] khi chạy script trực tiếp.
# Điều này khiến rl/config.py shadow config/ package của project → phải xóa.
_RL_DIR = str(Path(__file__).resolve().parent)
for _p in [_RL_DIR, _RL_DIR + "\\"]:
    while _p in sys.path:
        sys.path.remove(_p)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

print("Testing imports...")
from entities.boss_stage import get_stage_config, spawn_boss_bullets
print("[OK] boss_stage")

from entities.boss_ppo import BossPPO
print("[OK] boss_ppo")

from managers.boss_bullet_manager import BossBulletManager, BossBullet
print("[OK] boss_bullet_manager")

from rl.ppo_state_encoder import PPOStateEncoder
print("[OK] ppo_state_encoder")

from rl.ppo_rewards import PPORewardShaper
print("[OK] ppo_rewards")

from rl.ppo_config import PPOConfig
cfg = PPOConfig()
print(f"[OK] ppo_config  (n_envs={cfg.n_envs}, total_ts={cfg.total_timesteps:,})")

from core.game_ppo import GamePPO
print("[OK] game_ppo")

from rl.bullet_hell_env import BulletHellEnv
print("[OK] bullet_hell_env")

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
print("[OK] MaskablePPO + ActionMasker")

print()
print("--- Smoke-test: BossPPO stage configs ---")
for s in range(1, 5):
    boss = BossPPO(stage=s)
    assert boss.health == get_stage_config(s)["hp"], f"HP mismatch stage {s}"
    print(f"  Stage {s}: HP={boss.health}, moves={boss._moves}, shoot={boss._shoot_enabled}, pattern={boss._pattern}")

print()
print("--- Smoke-test: GamePPO init_match ---")
game = GamePPO(use_sim_time=True, sim_step_ms=16)
for s in range(1, 5):
    game.init_match(stage=s)
    assert game.boss is not None
    assert game.boss._stage == s
    print(f"  Stage {s}: boss.health={game.boss.health}")

print()
print("--- Smoke-test: PPOStateEncoder ---")
enc = PPOStateEncoder()
game.init_match(stage=2)
obs = enc.encode(game)
import numpy as np
assert obs.shape == (45,), f"Expected (45,) got {obs.shape}"
print(f"  obs.shape={obs.shape}  min={obs.min():.3f}  max={obs.max():.3f}")

print()
print("--- Smoke-test: BulletHellEnv (Stage 1, 50 steps) ---")
env = BulletHellEnv(render_mode=None, max_steps=200)
obs, info = env.reset()
stage = info["stage"]
print(f"  obs.shape={obs.shape}, stage={stage}")
assert obs.shape == (45,), f"Expected (45,) got {obs.shape}"

mask = env.action_masks()
print(f"  action_mask={mask}")

total_reward = 0.0
for step in range(50):
    action = env.action_space.sample()
    obs, rew, term, trunc, info = env.step(action)
    total_reward += rew
    if term or trunc:
        obs, info = env.reset()
        break

print(f"  50 steps OK, total_reward={total_reward:.3f}")
env.close()

print()
print("--- Smoke-test: ActionMasker wrapping ---")
env2 = BulletHellEnv(render_mode=None, max_steps=100)
masked_env = ActionMasker(env2, lambda e: e.env.action_masks())
obs, info = masked_env.env.reset()
print(f"  ActionMasker wrapping OK, obs.shape={obs.shape}")
masked_env.env.close()

print()
print("ALL CHECKS PASSED")
