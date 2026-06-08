"""
adaptive_callback.py — AdaptiveHyperparameterCallback cho MaskablePPO.

Tích hợp theo đặc tả kỹ thuật:
  - Adaptive Entropy Control: Tự động tăng ent_coef khi Policy bị bó cứng
  - Dynamic Learning Rate Tuning: Tăng LR khi Value Network ổn định & Phase >= 4
  - Auto-Backup: Gắn model_saver_callback vào AutomatedCurriculumWrapper
  - Kiểm tra định kỳ mỗi 10,000 steps

Yêu cầu:
  - stable_baselines3 >= 1.8
  - sb3_contrib (MaskablePPO)
  - eval_env phải là AutomatedCurriculumWrapper hoặc VecEnv bọc nó
"""

import os
from stable_baselines3.common.callbacks import BaseCallback

# Ngưỡng entropy_loss để kích hoạt Adaptive Entropy
# Nếu entropy > -0.20 (không đủ âm = Policy chưa đủ ngẫu nhiên) → tăng ent_coef
_ENTROPY_LOW_THRESHOLD = -0.20
_ENT_COEF_CEILING = 0.05        # Trần tuyệt đối cho ent_coef
_ENT_COEF_BOOST_FACTOR = 1.5   # Hệ số nhân khi phát hiện Policy bảo thủ

# Ngưỡng explained_variance để kích hoạt Dynamic LR
_EXP_VAR_THRESHOLD = 0.95
_LR_PHASE4_LOW = 5e-5
_LR_PHASE4_HIGH = 8e-5
_MIN_PHASE_FOR_LR_BOOST = 4     # Chỉ boost LR từ Phase 4 trở lên


class AdaptiveHyperparameterCallback(BaseCallback):
    """
    Callback SB3 tự động điều chỉnh siêu tham số trong khi chạy máy.

    Hai chế độ hoạt động (kiểm tra mỗi 10,000 steps):

    1. ADAPTIVE ENTROPY CONTROL
       Nếu entropy_loss > -0.20 (Policy bảo thủ):
       → Nhân ent_coef lên 1.5× (tối đa 0.05)
       → In thông báo ⚠️ [ADAPTIVE ENTROPY]

    2. DYNAMIC LEARNING RATE TUNING
       Nếu explained_variance > 0.95 VÀ Phase >= 4 VÀ LR đang ở 5e-5:
       → Đẩy LR lên 8e-5
       → In thông báo 🚀 [DYNAMIC LR]
    """

    def __init__(self, eval_env, save_dir: str = "./checkpoints", verbose: int = 0):
        """
        :param eval_env: AutomatedCurriculumWrapper hoặc VecEnv bọc nó
        :param save_dir: Thư mục lưu checkpoint tự động
        :param verbose: 0 = im lặng, 1 = log chi tiết
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Tracking
        self._entropy_boost_count: int = 0
        self._lr_boosted: bool = False
        self._last_phase: int = 1

        # Guard: _on_training_start chỉ chạy 1 lần dù model.learn() gọi nhiều lần
        # SB3 gọi _on_training_start() ở đầu mỗi model.learn() → phải chặn re-init
        self._initialized: bool = False

    def _on_training_start(self) -> None:
        """Gắn model_saver_callback vào AutomatedCurriculumWrapper sau khi model sẵn sàng."""
        if self._initialized:
            return  # Đã init rồi, bỏ qua — tránh reset _last_phase mỗi iteration
        self._initialized = True
        self._attach_saver_callback(self.eval_env)

    def _attach_saver_callback(self, env):
        """
        Duyệt qua lớp Wrapper để tìm AutomatedCurriculumWrapper và gắn callback.
        Hỗ trợ cả VecEnv (qua env_method) và Wrapper trực tiếp.
        """
        from rl.auto_curriculum_env import AutomatedCurriculumWrapper

        # Trường hợp 1: env chính là AutomatedCurriculumWrapper
        if isinstance(env, AutomatedCurriculumWrapper):
            env.model_saver_callback = self.save_model_checkpoint
            print(f"💾 [CALLBACK] Đã gắn Auto-Backup vào AutomatedCurriculumWrapper trực tiếp.")
            return

        # Trường hợp 2: VecEnv — dùng env_method (không thể gắn trực tiếp)
        # Phase Promotion sẽ được phát hiện qua polling trong _on_step
        if hasattr(env, "env_method"):
            print(
                f"💾 [CALLBACK] Auto-Backup sẵn sàng (VecEnv mode). "
                f"Phase Promotion sẽ kích hoạt backup tự động qua polling mỗi 10,000 steps."
            )
            return

        # Trường hợp 3: Wrapper lồng nhau — đệ quy tìm
        if hasattr(env, "env"):
            self._attach_saver_callback(env.env)

    # ------------------------------------------------------------------
    # Auto-Backup
    # ------------------------------------------------------------------

    def save_model_checkpoint(self, filename: str):
        """Lưu model ra file .zip trong thư mục checkpoints."""
        save_path = os.path.join(self.save_dir, f"{filename}.zip")
        self.model.save(save_path)
        print(f"💾 [HỆ THỐNG] Đã tự động sao lưu Checkpoint an toàn tại: {save_path}")

    # ------------------------------------------------------------------
    # _on_step — kiểm tra định kỳ mỗi 10,000 steps
    # ------------------------------------------------------------------

    def _on_step(self) -> bool:
        if self.n_calls % 10000 != 0:
            return True

        # Lấy entropy_loss và explained_variance từ Tensorboard Logger
        current_entropy = self.logger.name_to_value.get("train/entropy_loss", None)
        exp_var = self.logger.name_to_value.get("train/explained_variance", 0.0)

        # Xác định Phase hiện tại (từ eval_env hoặc training env)
        current_phase = self._get_current_phase()

        # --- CHẾ ĐỘ 1: ADAPTIVE ENTROPY CONTROL ---
        if current_entropy is not None and current_entropy != -1.0:
            if current_entropy > _ENTROPY_LOW_THRESHOLD:
                old_ent = float(self.model.ent_coef)
                new_ent = min(old_ent * _ENT_COEF_BOOST_FACTOR, _ENT_COEF_CEILING)
                self.model.ent_coef = new_ent
                self._entropy_boost_count += 1
                print(
                    f"⚠️ [ADAPTIVE ENTROPY] Phát hiện mạng Policy quá bảo thủ "
                    f"(entropy_loss={current_entropy:.4f} > {_ENTROPY_LOW_THRESHOLD}). "
                    f"Tăng ent_coef: {old_ent:.5f} → {new_ent:.5f} "
                    f"(lần #{self._entropy_boost_count})"
                )

        # --- CHẾ ĐỘ 2: DYNAMIC LEARNING RATE TUNING ---
        if exp_var > _EXP_VAR_THRESHOLD and current_phase >= _MIN_PHASE_FOR_LR_BOOST:
            current_lr = self._get_current_lr()
            if abs(current_lr - _LR_PHASE4_LOW) < 1e-8 and not self._lr_boosted:
                self._set_lr(_LR_PHASE4_HIGH)
                self._lr_boosted = True
                print(
                    f"🚀 [DYNAMIC LR] Mạng Value ổn định (exp_var={exp_var:.3f} > {_EXP_VAR_THRESHOLD}) "
                    f"tại Phase {current_phase}. "
                    f"Đẩy nhanh tốc độ học LR: {_LR_PHASE4_LOW:.0e} → {_LR_PHASE4_HIGH:.0e}"
                )

        # --- POLLING AUTO-BACKUP cho VecEnv ---
        self._poll_phase_and_backup(current_phase)

        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_current_phase(self) -> int:
        """Lấy train_phase từ eval_env (AutomatedCurriculumWrapper hoặc VecEnv)."""
        from rl.auto_curriculum_env import AutomatedCurriculumWrapper

        if isinstance(self.eval_env, AutomatedCurriculumWrapper):
            return self.eval_env.train_phase

        # VecEnv: dùng env_method để lấy attribute từ sub-env đầu tiên
        if hasattr(self.eval_env, "env_method"):
            try:
                results = self.eval_env.env_method("get_phase_info")
                if results:
                    return results[0].get("train_phase", 1)
            except Exception:
                pass

        # Training env (self.training_env)
        if hasattr(self.training_env, "env_method"):
            try:
                results = self.training_env.env_method("get_phase_info")
                if results:
                    return results[0].get("train_phase", 1)
            except Exception:
                pass

        return getattr(self, "_last_phase", 1)

    def _poll_phase_and_backup(self, current_phase: int):
        """Phát hiện thăng Phase qua polling (dành cho trường hợp VecEnv)."""
        last_phase = getattr(self, "_last_phase", 1)
        if current_phase > last_phase:
            self.save_model_checkpoint(f"checkpoint_phase_{last_phase}_completed")
            print(
                f"🔥 [CALLBACK] Phát hiện thăng Phase {last_phase} → {current_phase}. "
                f"Đã kích hoạt Auto-Backup!"
            )
        self._last_phase = current_phase

    def _get_current_lr(self) -> float:
        """Lấy learning rate hiện tại từ optimizer của model."""
        try:
            return self.model.policy.optimizer.param_groups[0]["lr"]
        except (AttributeError, IndexError):
            return float(getattr(self.model, "learning_rate", 5e-5))

    def _set_lr(self, lr: float):
        """Cập nhật learning rate trong cả lr_schedule và optimizer."""
        self.model.lr_schedule = lambda _: lr
        for param_group in self.model.policy.optimizer.param_groups:
            param_group["lr"] = lr
