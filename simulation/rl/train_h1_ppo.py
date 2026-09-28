#!/usr/bin/env python3
"""
سكربت تدريب الروبوت H1 على المشي + العقبات
==============================================
Curriculum Training Pipeline:
  Stage 0: مشي على أرض مسطحة (أبسط حاجة)
  Stage 1: مشي + حفرة
  Stage 2: مشي + حفرة + بوكس

كل مرحلة بندرب عليها لحد ما الروبوت يعرف يعديها
وبعدين ننقل للمرحلة اللي بعدها.
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Fix Windows Unicode encoding issue
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from h1_locomotion_env import H1ObstacleCourseEnv


class TrainingMonitor(BaseCallback):
    """
    Callback بيتابع التدريب ويطبع معلومات مفيدة كل شوية.
    بيحفظ الموديل كل فترة (checkpoint) عشان لو حصل حاجة منخسرش الشغل.
    """
    def __init__(self, save_dir: str, save_every: int = 50000, verbose: int = 1):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.save_every = save_every
        self.best_reward = -float("inf")
        self._last_save = 0

    def _on_step(self) -> bool:
        # حفظ checkpoint كل فترة
        if self.num_timesteps - self._last_save >= self.save_every:
            self._last_save = self.num_timesteps
            path = os.path.join(self.save_dir, f"checkpoint_{self.num_timesteps}.zip")
            self.model.save(path)
            if self.verbose > 0:
                print(f"\n  [CHECKPOINT] Saved at step {self.num_timesteps:,}")
        return True


def make_env(stage: int, use_obstacles: bool):
    """Factory function لعمل بيئة حسب المرحلة"""
    def _init():
        return H1ObstacleCourseEnv(
            use_obstacles=use_obstacles,
            curriculum_stage=stage,
        )
    return _init


def train_stage(
    stage: int,
    total_steps: int,
    n_envs: int,
    save_dir: str,
    load_from: str = None,
):
    """
    تدريب مرحلة واحدة من المنهج
    
    stage: رقم المرحلة (0, 1, 2)
    total_steps: عدد الخطوات الكلية
    n_envs: عدد البيئات المتوازية
    save_dir: مكان حفظ الموديل
    load_from: مسار موديل قديم نكمل عليه
    """
    stage_names = {
        0: "مشي مستقيم على أرض مسطحة (Flat Walking)",
        1: "مشي + حفرة (Walking + Pit)",
        2: "مشي + حفرة + بوكس (Walking + Pit + Box)",
    }
    
    use_obstacles = (stage > 0)
    
    print("\n" + "=" * 65)
    print(f"  STAGE {stage}: {stage_names.get(stage, 'Unknown')}")
    print(f"  Steps: {total_steps:,} | Workers: {n_envs}")
    if load_from:
        print(f"  Continuing from: {load_from}")
    print("=" * 65 + "\n")

    # عمل بيئات متوازية
    env = make_vec_env(make_env(stage, use_obstacles), n_envs=n_envs)

    # PPO hyperparameters محسنة للمشي
    if load_from and os.path.exists(load_from):
        print(f"[INFO] Loading existing model from {load_from}")
        model = PPO.load(load_from, env=env)
        # learning rate أقل لما بنكمل تدريب
        model.learning_rate = 2e-4
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            batch_size=256,         # أكبر = أثبت
            n_steps=4096,           # أطول = أحسن لتقدير المكافأة
            gamma=0.99,             # خصم المكافأة المستقبلية
            gae_lambda=0.95,        # تقدير المزايا العامة
            clip_range=0.2,         # نطاق القطع
            ent_coef=0.01,          # معامل الـ entropy (أعلى = أكتر استكشاف)
            vf_coef=0.5,            # معامل الـ value function
            max_grad_norm=0.5,      # تقييد الـ gradient
            n_epochs=10,            # عدد مرات التحديث لكل batch
            tensorboard_log=str(Path(save_dir) / "tb_logs"),
            policy_kwargs={
                "net_arch": [256, 256],  # شبكة أكبر = قدرة تعلم أعلى
            },
        )

    # Callback للمتابعة والحفظ
    monitor = TrainingMonitor(save_dir=save_dir, save_every=100000)

    # بدء التدريب
    start_time = time.time()
    try:
        model.learn(total_timesteps=total_steps, progress_bar=True, callback=monitor)
    except KeyboardInterrupt:
        print("\n[WARN] Interrupted! Saving emergency checkpoint...")

    elapsed = time.time() - start_time
    print(f"\n[TIME] Stage {stage} took {elapsed/60:.1f} minutes")

    # حفظ النتيجة النهائية
    stage_path = str(Path(save_dir) / f"h1_stage{stage}_policy.zip")
    model.save(stage_path)
    print(f"[SUCCESS] Stage {stage} model saved to: {stage_path}")

    # حفظ نسخة عامة كمان (h1_walking_policy.zip)
    general_path = str(Path(save_dir) / "h1_walking_policy.zip")
    model.save(general_path)
    print(f"[SUCCESS] Also saved as: {general_path}")

    env.close()
    return stage_path


def main():
    parser = argparse.ArgumentParser(description="H1 Curriculum Training")
    parser.add_argument("--stage", type=int, default=0, choices=[0, 1, 2],
                        help="Training stage (0=walk, 1=pit, 2=box)")
    parser.add_argument("--steps", type=int, default=2_000_000,
                        help="Total timesteps per stage")
    parser.add_argument("--envs", type=int, default=4,
                        help="Parallel environments")
    parser.add_argument("--continue-from", type=str, default=None,
                        help="Path to model to continue training from")
    parser.add_argument("--all-stages", action="store_true",
                        help="Train all stages sequentially")
    args = parser.parse_args()

    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 65)
    print("    ANSA OS - Unitree H1 Curriculum Training")
    print("    المنهج التدريبي: مشي → حفرة → بوكس")
    print("=" * 65)

    if args.all_stages:
        # تدريب كل المراحل بالترتيب
        model_path = args.continue_from
        for stage in range(3):
            steps = args.steps if stage == 0 else int(args.steps * 0.75)
            model_path = train_stage(
                stage=stage,
                total_steps=steps,
                n_envs=args.envs,
                save_dir=save_dir,
                load_from=model_path,
            )
        print("\n" + "=" * 65)
        print("  ALL STAGES COMPLETE!")
        print("  Run: python evaluate_h1_policy.py --stage 2")
        print("=" * 65)
    else:
        # تدريب مرحلة واحدة بس
        train_stage(
            stage=args.stage,
            total_steps=args.steps,
            n_envs=args.envs,
            save_dir=save_dir,
            load_from=args.continue_from,
        )
        print(f"\n  Run: python evaluate_h1_policy.py --stage {args.stage}")


if __name__ == "__main__":
    main()
