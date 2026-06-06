"""
train_ppo.py — Vòng lặp huấn luyện MaskablePPO với Robust Auto-Curriculum Learning.

Triển khai theo spec void_survivor.md (Mục II, III):
  - Stage Mixing 70/20/10 tại env.reset()
  - Decoupled Evaluation mỗi PPO_EVAL_INTERVAL steps
  - Điều kiện nâng Stage: win_rate > 80% & old stages không sụt > 5%
  - Automatic Rollback: win_rate < 15% liên tục 3 kỳ eval
  - Dynamic Entropy Schedule: tăng ent_coef khi lên stage mới, giảm dần về floor
"""

# ---------------------------------------------------------------------------
# sys.path guard — phải đặt TRƯỚC tất cả import nội bộ.
# Khi chạy `python rl/train_ppo.py`, Python tự động thêm rl/ vào sys.path[0],
# khiến rl/config.py shadow config/ package của project.
# ---------------------------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
if hasattr(_sys.stdout, 'reconfigure'):
    _sys.stdout.reconfigure(encoding='utf-8')

_ROOT = str(_Path(__file__).resolve().parents[1])
_RL_DIR = str(_Path(__file__).resolve().parent)
for _p in [_RL_DIR, _RL_DIR + "\\"]:
    while _p in _sys.path:
        _sys.path.remove(_p)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# Standard imports
# ---------------------------------------------------------------------------
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# SB3 + SB3-Contrib
try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecEnv
except ImportError:
    print(
        "[ERROR] Thiếu thư viện. Cài đặt bằng lệnh:\n"
        "  pip install sb3-contrib stable-baselines3"
    )
    sys.exit(1)

from rl.bullet_hell_env import BulletHellEnv
from rl.ppo_config import PPOConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_run_dir(cfg: PPOConfig) -> Path:
    name = cfg.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(cfg.models_dir) / name


def _make_env_fn(cfg: PPOConfig, render: bool = False):
    """Factory tạo BulletHellEnv đã bọc ActionMasker."""
    def _factory():
        env = BulletHellEnv(
            render_mode="human" if render else None,
            max_steps=cfg.max_steps,
            render_fps=cfg.render_fps,
        )
        env = ActionMasker(env, lambda e: e.action_masks())
        return env
    return _factory


def _set_stage_all_envs(vec_env: VecEnv, stage: int):
    """Đồng bộ current_stage xuống tất cả sub-environments."""
    vec_env.env_method("set_stage", stage)


def _write_meta(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Decoupled Evaluation
# ---------------------------------------------------------------------------

def evaluate_stage(model: MaskablePPO, stage: int, cfg: PPOConfig) -> float:
    """
    Chạy cfg.eval_episodes trận độc lập trên stage cụ thể.
    Trả về win_rate [0, 1].
    """
    from rl.ppo_state_encoder import PPOStateEncoder
    enc = PPOStateEncoder()
    wins = 0

    for _ in range(cfg.eval_episodes):
        env = BulletHellEnv(max_steps=cfg.max_steps)
        # Ép stage cụ thể — bỏ qua stage mixing
        env.current_stage = stage
        env._active_stage = stage
        env.steps_in_stage = 999999  # Đánh giá ở mức độ khó tối đa (không bảo hiểm/DDA)
        env.game.init_match(stage=stage, stage_steps=999999)
        obs = enc.encode(env.game)
        env._reward_shaper.reset(env.game)
        env._steps = 0

        done = False
        info = {}
        while not done:
            # Truyền action_masks để MaskablePPO không chọn action bị khóa (vd: bắn khi cooldown)
            masks = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=masks)
            obs, _rew, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

        if info.get("is_win", False):
            wins += 1
        env.close()

    return wins / max(1, cfg.eval_episodes)


def evaluate_all_stages(
    model: MaskablePPO,
    current_stage: int,
    cfg: PPOConfig,
    prev_win_rates: dict,
) -> tuple[float, bool, dict[int, float]]:
    """
    Đánh giá tất cả stage từ 1 → current_stage.
    Trả về (win_rate_current, is_old_stage_corrupted, current_win_rates).
    """
    win_rate_current = 0.0
    is_corrupted = False
    current_win_rates = {}

    for s in range(1, current_stage + 1):
        wr = evaluate_stage(model, s, cfg)
        print(f"  [Eval] Stage {s}: win_rate = {wr:.1%}")
        current_win_rates[s] = wr

        if s == current_stage:
            win_rate_current = wr
        else:
            prev = prev_win_rates.get(s, wr)
            if (prev - wr) > cfg.old_stage_drop_limit:
                print(f"  [WARN] Stage {s} sụt từ {prev:.1%} → {wr:.1%} (>{cfg.old_stage_drop_limit:.0%})")
                is_corrupted = True

        prev_win_rates[s] = max(prev_win_rates.get(s, 0.0), wr)

    return win_rate_current, is_corrupted, current_win_rates


# ---------------------------------------------------------------------------
# Dynamic Entropy Schedule
# ---------------------------------------------------------------------------

class EntropyScheduler:
    """
    Khi lên Stage mới: tăng ent_coef thêm boost trong boost_steps đầu.
    Sau đó giảm dần tuyến tính về floor.
    """

    def __init__(self, cfg: PPOConfig):
        self._base = cfg.ent_coef
        self._boost = cfg.ent_coef_boost
        self._boost_steps = cfg.ent_coef_boost_steps
        self._floor = cfg.ent_coef_floor
        self._stage_up_at: int | None = None
        self._current = cfg.ent_coef

    def on_stage_up(self, global_step: int):
        self._stage_up_at = global_step
        self._current = self._base + self._boost
        print(f"  [Entropy] Tăng ent_coef → {self._current:.4f} (boost {self._boost_steps} steps)")

    def get(self, global_step: int) -> float:
        if self._stage_up_at is None:
            return self._current
        elapsed = global_step - self._stage_up_at
        if elapsed >= self._boost_steps:
            self._current = self._floor
            self._stage_up_at = None
        else:
            progress = elapsed / self._boost_steps
            self._current = (self._base + self._boost) * (1 - progress) + self._floor * progress
        return self._current


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: PPOConfig | None = None, resume_path: str | None = None, start_stage: int = 1):
    cfg = cfg or PPOConfig()
    run_dir = _build_run_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = run_dir / "best_model.zip"
    meta_path = run_dir / "meta.json"

    print(f"[PPO] Run dir: {run_dir}")
    print(f"[PPO] Total timesteps: {cfg.total_timesteps:,}")
    print(f"[PPO] n_envs: {cfg.n_envs}  |  eval_interval: {cfg.eval_interval:,}")

    # Vectorized Environment
    vec_env = make_vec_env(_make_env_fn(cfg, render=False), n_envs=cfg.n_envs)

    # MaskablePPO model — tải lại nếu có resume_path, ngược lại tạo mới
    if resume_path:
        print(f"[Resume] Đang nạp checkpoint: {resume_path}")
        model = MaskablePPO.load(resume_path, env=vec_env)
        # Giữ nguyên LR từ config để tiếp tục train ổn định
        for pg in model.policy.optimizer.param_groups:
            pg["lr"] = cfg.learning_rate
        print(f"[Resume] Đặt lại lr = {cfg.learning_rate}")
    else:
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            learning_rate=cfg.learning_rate,
            gamma=cfg.gamma,
            ent_coef=cfg.ent_coef,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            clip_range=cfg.clip_range,
            verbose=1,
            tensorboard_log=cfg.tensorboard_log,
        )

    # Curriculum state
    current_stage = max(1, min(5, start_stage))
    _set_stage_all_envs(vec_env, current_stage)
    if resume_path:
        print(f"[Resume] Bắt đầu từ Stage {current_stage}\n")

    entropy_sched = EntropyScheduler(cfg)
    prev_win_rates: dict[int, float] = {}
    
    # Best model tracking
    best_stage = 0
    best_avg_win_rate = -1.0
    best_prev_win_rates: dict[int, float] = {}
    
    rollback_fail_count = 0
    steps_trained = 0
    learn_block = cfg.eval_interval

    iterations = cfg.total_timesteps // learn_block
    print(f"[PPO] Sẽ chạy {iterations} iteration × {learn_block:,} steps/iter\n")

    for iteration in range(1, iterations + 1):

        # Dynamic entropy
        current_ent = entropy_sched.get(steps_trained)
        model.ent_coef = current_ent

        # Train block
        model.learn(total_timesteps=learn_block, reset_num_timesteps=False, progress_bar=False)
        steps_trained += learn_block

        print(f"\n{'='*60}")
        print(f"[Iter {iteration}/{iterations}] Steps: {steps_trained:,} | Stage: {current_stage} | ent_coef: {current_ent:.4f}")

        # Decoupled Evaluation
        print("[Eval] Đang đánh giá độc lập...")
        win_rate_cur, is_corrupted, current_win_rates = evaluate_all_stages(model, current_stage, cfg, prev_win_rates)
        print(f"[Eval] Stage {current_stage} win_rate = {win_rate_cur:.1%} | corrupted = {is_corrupted}")

        # Lưu best model
        current_avg_wr = float(np.mean(list(current_win_rates.values()))) if current_win_rates else 0.0
        
        # Tiêu chí lưu best model:
        # 1. Đạt stage cao hơn best_stage đã lưu trước đó.
        # 2. Hoặc cùng stage nhưng avg_win_rate hiện tại cao hơn.
        # Đồng thời mô hình không được bị corrupted (sụt giảm quá nhiều ở các stage cũ).
        is_new_best = False
        if not is_corrupted:
            if current_stage > best_stage:
                is_new_best = True
            elif current_stage == best_stage and current_avg_wr > best_avg_win_rate:
                is_new_best = True

        if is_new_best:
            best_stage = current_stage
            best_avg_win_rate = current_avg_wr
            best_prev_win_rates = prev_win_rates.copy()
            model.save(str(best_model_path))
            print(f"[Checkpoint] Best model → {best_model_path} (Stage: {best_stage}, avg_wr={best_avg_win_rate:.1%})")

        # Nâng Stage
        if win_rate_cur > cfg.promote_threshold and not is_corrupted and current_stage < 5:
            current_stage += 1
            _set_stage_all_envs(vec_env, current_stage)
            entropy_sched.on_stage_up(steps_trained)
            model.ent_coef = entropy_sched.get(steps_trained)
            rollback_fail_count = 0
            
            # Giảm Learning Rate khi chuyển sang stage mới để tránh sập policy (3e-4 -> 1e-4)
            current_lr = 1e-4
            for param_group in model.policy.optimizer.param_groups:
                param_group['lr'] = current_lr
            print(f"  [Learning Rate] Đã giảm lr → {current_lr} để ổn định học stage mới")
            
            print(f"\n{'*'*60}\n  *** TIẾN LÊN STAGE {current_stage}! ***\n{'*'*60}\n")
            model.save(str(run_dir / f"stage_{current_stage}_entry.zip"))

        # Rollback
        elif win_rate_cur < cfg.rollback_threshold and current_stage > 1:
            rollback_fail_count += 1
            print(f"[WARN] Rollback counter: {rollback_fail_count}/{cfg.rollback_patience}")
            if rollback_fail_count >= cfg.rollback_patience:
                print("\n!!! PHÁT HIỆN SẬP POLICY — KÍCH HOẠT ROLLBACK !!!")
                if best_model_path.exists():
                    model = MaskablePPO.load(str(best_model_path), env=vec_env)
                    print(f"[Rollback] Đã tải lại {best_model_path}")
                # Reset curriculum state
                current_stage = max(1, best_stage)
                prev_win_rates = best_prev_win_rates.copy()
                _set_stage_all_envs(vec_env, current_stage)
                rollback_fail_count = 0
                
                # Khôi phục Learning Rate mặc định khi quay lại stage cũ
                current_lr = cfg.learning_rate
                for param_group in model.policy.optimizer.param_groups:
                    param_group['lr'] = current_lr
                print(f"[Rollback] Quay lại Stage {current_stage} | Khôi phục lr → {current_lr}\n")
        else:
            rollback_fail_count = 0

        _write_meta(meta_path, {
            "iteration": iteration,
            "steps_trained": steps_trained,
            "current_stage": current_stage,
            "win_rate_current_stage": round(win_rate_cur, 4),
            "best_stage": best_stage,
            "best_avg_win_rate": round(best_avg_win_rate, 4),
            "ent_coef": round(current_ent, 6),
            "rollback_fail_count": rollback_fail_count,
        })

    final_path = run_dir / "final_model.zip"
    model.save(str(final_path))
    print(f"\n[Done] Final model → {final_path}")
    vec_env.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MaskablePPO — Void Survivor")
    parser.add_argument("--timesteps",   type=int,  default=None, help="Override PPO_TOTAL_TIMESTEPS")
    parser.add_argument("--envs",        type=int,  default=None, help="Override PPO_N_ENVS")
    parser.add_argument("--run-name",    type=str,  default=None, help="Override PPO_RUN_NAME")
    parser.add_argument("--render",      action="store_true",     help="Bật render (chậm hơn)")
    parser.add_argument("--resume",      type=str,  default=None,
                        help="Đường dẫn file .zip cần nạp lại (ví dụ: models/ppo/xxx/stage_4_entry.zip)")
    parser.add_argument("--start-stage", type=int,  default=1,
                        help="Stage bắt đầu khi resume (mặc định: 1)")
    args = parser.parse_args()

    cfg = PPOConfig()
    if args.timesteps:
        cfg.total_timesteps = args.timesteps
    if args.envs:
        cfg.n_envs = args.envs
    if args.run_name:
        cfg.run_name = args.run_name
    if args.render:
        cfg.render = True

    train(cfg, resume_path=args.resume, start_stage=args.start_stage)
