"""
train_ppo.py — Vòng lặp huấn luyện MaskablePPO với Automated Curriculum Learning.

Tổ chức theo cấu trúc mô đun sạch:
  - AutomatedCurriculumWrapper: Điều khiển phân phối Stage theo 5 Phase tự động
  - AdaptiveHyperparameterCallback: Điều chỉnh Entropy + LR theo thời gian thực
  - Decoupled Evaluation mỗi PPO_EVAL_INTERVAL steps
  - train_ppo chỉ còn: train → eval → sync Rolling Window → lưu best model → ghi meta
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
from rl.auto_curriculum_env import AutomatedCurriculumWrapper
from rl.adaptive_callback import AdaptiveHyperparameterCallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_run_dir(cfg: PPOConfig) -> Path:
    name = cfg.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(cfg.models_dir) / name


def _make_env_fn(cfg: PPOConfig, render: bool = False, use_curriculum_wrapper: bool = True):
    """Factory tạo BulletHellEnv đã bọc ActionMasker (và tùy chọn AutomatedCurriculumWrapper)."""
    def _factory():
        env = BulletHellEnv(
            render_mode="human" if render else None,
            max_steps=cfg.max_steps,
            render_fps=cfg.render_fps,
        )
        if use_curriculum_wrapper:
            env = AutomatedCurriculumWrapper(env)
        env = ActionMasker(env, lambda e: e.action_masks())
        return env
    return _factory



def _write_meta(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Hằng số Learning Rate và mapping Phase → Eval Stage
# ---------------------------------------------------------------------------
_DEFAULT_LR = 5e-5  # LR ổn định, AdaptiveCallback sẽ tự boost khi cần

# Phase hiện tại → Stage cao nhất cần eval để đo lường đúng tiến độ
# Phase 1: eval đến Stage 2 (Stage 3 chỉ là preview, chưa yêu cầu thắng)
# Phase 4/5: eval tất cả đến Stage 6
_PHASE_TO_EVAL_STAGE: dict[int, int] = {
    1: 2,
    2: 4,
    3: 5,
    4: 6,
    5: 6,
}


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




def set_model_learning_rate(model: MaskablePPO, lr: float):
    """Cập nhật learning rate cả trong PyTorch optimizer lẫn lr_schedule của SB3."""
    model.lr_schedule = lambda _: lr
    for param_group in model.policy.optimizer.param_groups:
        param_group["lr"] = lr


# ---------------------------------------------------------------------------
# Hàm đồng bộ Rolling Window (tách khỏi vòng lặp — fix #8)
# ---------------------------------------------------------------------------

def _sync_eval_to_rolling_window(vec_env, stage: int, win_rate: float, n_episodes: int):
    """
    Nạp kết quả eval vào stage_history của AutomatedCurriculumWrapper trong mọi sub-env.
    ActionMasker sẽ delegate update_post_episode / check_phase_promotion xuống Wrapper.
    """
    wins = round(win_rate * n_episodes)
    results_bool = [True] * wins + [False] * (n_episodes - wins)
    try:
        for is_win in results_bool:
            vec_env.env_method("update_post_episode", stage, is_win)
        vec_env.env_method("check_phase_promotion")
    except Exception as e:
        msg = str(e)
        if "update_post_episode" not in msg and "check_phase_promotion" not in msg:
            print(f"  [WARN] Không thể sync Rolling Window stage {stage}: {e}")


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
        set_model_learning_rate(model, _DEFAULT_LR)
        print(f"[Resume] Đặt lại lr = {_DEFAULT_LR} (AdaptiveCallback sẽ tự boost sau)")
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
        set_model_learning_rate(model, _DEFAULT_LR)

    # current_stage chỉ dùng để tracking và eval — AutomatedCurriculumWrapper tự chọn Stage khi reset()
    current_stage = max(1, min(6, start_stage))
    if resume_path:
        print(f"[Resume] Bắt đầu từ Stage {current_stage} (AutomatedCurriculumWrapper sẽ tự phân phối)\n")

    # State theo dõi — best model tracking
    best_stage = 0
    best_avg_win_rate = -1.0
    steps_trained = 0
    learn_block = cfg.eval_interval

    iterations = cfg.total_timesteps // learn_block
    print(f"[PPO] Sẽ chạy {iterations} iteration × {learn_block:,} steps/iter\n")

    # --- Khởi tạo AdaptiveHyperparameterCallback ---
    run_dir_str = str(run_dir / "checkpoints")
    adaptive_callback = AdaptiveHyperparameterCallback(
        eval_env=vec_env,
        save_dir=run_dir_str,
        verbose=1,
    )
    print(f"[PPO] AdaptiveHyperparameterCallback sẵn sàng. Auto-backup tại: {run_dir_str}")


    # === VÒNG LẶP HUẤN LUYỆN — AutomatedCurriculumWrapper điều khiển Stage ===
    # AdaptiveHyperparameterCallback điều khiển Entropy + LR theo thời gian thực.
    # train_ppo chỉ còn: train → eval → sync Rolling Window → lưu best model → ghi meta.
    prev_win_rates: dict[int, float] = {}
    for iteration in range(1, iterations + 1):

        # Lấy ent_coef hiện tại để log (AdaptiveCallback đã tự điều chỉnh bên trong)
        current_ent = float(model.ent_coef)

        # Train block — AdaptiveHyperparameterCallback chạy bên trong
        model.learn(
            total_timesteps=learn_block,
            reset_num_timesteps=False,
            progress_bar=False,
            callback=adaptive_callback,
        )
        steps_trained += learn_block

        print(f"\n{'='*60}")
        print(f"[Iter {iteration}/{iterations}] Steps: {steps_trained:,} | ent_coef: {current_ent:.4f}")

        # === Eval và sync Rolling Window ===
        print("[Eval] Đang đánh giá độc lập...")
        win_rate_cur, is_corrupted, current_win_rates = evaluate_all_stages(
            model, current_stage, cfg, prev_win_rates
        )
        print(f"[Eval] Stage {current_stage} win_rate = {win_rate_cur:.1%} | corrupted = {is_corrupted}")

        for s, wr in current_win_rates.items():
            _sync_eval_to_rolling_window(vec_env, s, wr, cfg.eval_episodes)

        # Lấy Phase hiện tại sau khi sync (có thể đã thăng Phase)
        try:
            phase_infos = vec_env.env_method("get_phase_info")
            current_train_phase = phase_infos[0].get("train_phase", 1) if phase_infos else 1
        except Exception:
            current_train_phase = 1

        # Cập nhật current_stage theo Phase mới — fix #4 (current_stage đóng băng)
        # current_stage được dùng cho eval vòng tiếp theo và lưu best model
        new_eval_stage = _PHASE_TO_EVAL_STAGE.get(current_train_phase, current_stage)
        if new_eval_stage > current_stage:
            print(f"  [Phase {current_train_phase}] Eval stage nâng lên đến Stage {new_eval_stage}")
            current_stage = new_eval_stage

        # Lưu Best Model
        current_avg_wr = float(np.mean(list(current_win_rates.values()))) if current_win_rates else 0.0

        is_new_best = False
        if not is_corrupted:
            if current_stage > best_stage:
                is_new_best = True
            elif current_stage == best_stage and current_avg_wr > best_avg_win_rate:
                is_new_best = True

        if is_new_best:
            best_stage = current_stage
            best_avg_win_rate = current_avg_wr
            model.save(str(best_model_path))
            print(f"[Checkpoint] Best model → {best_model_path} (Stage: {best_stage}, avg_wr={best_avg_win_rate:.1%})")

        _write_meta(meta_path, {
            "iteration": iteration,
            "steps_trained": steps_trained,
            "current_stage": current_stage,
            "train_phase": current_train_phase,
            "win_rate_current_stage": round(win_rate_cur, 4),
            "win_rates_all": {str(s): round(r, 4) for s, r in current_win_rates.items()},
            "best_stage": best_stage,
            "best_avg_win_rate": round(best_avg_win_rate, 4),
            "ent_coef": round(current_ent, 6),
            "is_corrupted": is_corrupted,
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
