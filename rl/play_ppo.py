"""
play_ppo.py — Xem agent thi đấu bằng mô hình MaskablePPO đã huấn luyện.
"""

import argparse
import sys
import time
from pathlib import Path

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
except ImportError:
    print("[ERROR] Cần cài đặt sb3-contrib: pip install sb3-contrib")
    sys.exit(1)

# Fix sys.path để tránh conflict rl/config.py
_ROOT = str(Path(__file__).resolve().parents[1])
_RL_DIR = str(Path(__file__).resolve().parent)
for _p in [_RL_DIR, _RL_DIR + "\\"]:
    while _p in sys.path:
        sys.path.remove(_p)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rl.bullet_hell_env import BulletHellEnv
from rl.ppo_state_encoder import PPOStateEncoder


def play(model_path: str, stage: int, episodes: int = 3, fps: int = 60):
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"[ERROR] Không tìm thấy model tại: {model_file}")
        sys.exit(1)

    print(f"Loading model: {model_file}")
    model = MaskablePPO.load(str(model_file))

    # Khởi tạo môi trường có render
    env = BulletHellEnv(render_mode="human", render_fps=fps)
    enc = PPOStateEncoder()

    for ep in range(episodes):
        print(f"\n--- Episode {ep + 1}/{episodes} (Stage {stage}) ---")
        
        # Bỏ qua stage mixing, ép chạy stage người dùng chỉ định
        env.current_stage = stage
        env._active_stage = stage
        env.game.init_match(stage=stage)
        
        obs = enc.encode(env.game)
        env._reward_shaper.reset(env.game)
        env._steps = 0
        
        done = False
        total_reward = 0.0

        while not done:
            # MaskablePPO predict
            action_masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            done = terminated or truncated

        win_str = "THẮNG" if info.get("is_win") else "THUA"
        print(f"Kết quả: {win_str} | Total Reward: {total_reward:.2f}")
        time.sleep(1)

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Đường dẫn đến file .zip (VD: models/ppo/run_1/best_model.zip)")
    parser.add_argument("--stage", type=int, default=1, help="Chạy ở Stage nào (1-6). Mặc định: 1")
    parser.add_argument("--episodes", type=int, default=3, help="Số trận muốn xem. Mặc định: 3")
    parser.add_argument("--fps", type=int, default=60, help="Tốc độ khung hình. Mặc định: 60")
    
    args = parser.parse_args()
    play(args.model_path, args.stage, args.episodes, args.fps)
