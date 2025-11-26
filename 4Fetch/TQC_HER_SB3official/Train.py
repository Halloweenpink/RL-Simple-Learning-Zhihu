import os
import time

import gymnasium as gym
import gymnasium_robotics

from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# 关键：TQC 在 sb3_contrib 里
from sb3_contrib import TQC

# -----------------------
# 超参（对齐 rl-zoo 的 HER+TQC FetchPnP）
# -----------------------
ENV_ID = "FetchPickAndPlace-v4"
TOTAL_TIMESTEPS = 1_000_000  # zoo 用 1e6
SEED = 42

N_SAMPLED_GOAL = 4
GOAL_SELECTION_STRATEGY = "future"

BATCH_SIZE = 2048           # zoo: 2048
BUFFER_SIZE = 1_000_000
LEARNING_RATE = 1e-3
GAMMA = 0.95
TAU = 0.05

TRAIN_FREQ = 1
GRADIENT_STEPS = 1
LEARNING_STARTS = 1000      # 和 zoo 一致

EVAL_FREQ = 10_000          # 评估频率可以稍微放大一点
N_EVAL_EPISODES = 15
CHECKPOINT_FREQ = 50_000

LOG_ROOT = "logs_pnp_tqc_her"
TB_LOGDIR = os.path.join(LOG_ROOT, "tb")
BEST_MODEL_DIR = os.path.join(LOG_ROOT, "best")
CKPT_DIR = os.path.join(LOG_ROOT, "checkpoints")


def main():
    os.makedirs(LOG_ROOT, exist_ok=True)
    os.makedirs(TB_LOGDIR, exist_ok=True)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    # 注册 Fetch 系列环境
    gym.register_envs(gymnasium_robotics)

    # 注意：off-policy + HER 建议单环境（n_envs=1）
    train_env = make_vec_env(
        ENV_ID,
        n_envs=1,
        env_kwargs={"render_mode": None},
        seed=SEED,
    )

    eval_env = make_vec_env(
        ENV_ID,
        n_envs=1,
        env_kwargs={"render_mode": None},
        seed=SEED + 1,
    )

    # TQC + HER 模型
    model = TQC(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
        train_freq=TRAIN_FREQ,
        gradient_steps=GRADIENT_STEPS,
        learning_starts=LEARNING_STARTS,
        # 这里直接用 zoo 的配置
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=N_SAMPLED_GOAL,
            goal_selection_strategy=GOAL_SELECTION_STRATEGY,
        ),
        policy_kwargs=dict(
            net_arch=[512, 512, 512],
            n_critics=2,
        ),
        tensorboard_log=TB_LOGDIR,
        seed=SEED,
        verbose=1,
        device="cuda",
        # top_quantiles_to_drop_per_net 默认 2，先不用动
    )

    # 评估回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=os.path.join(LOG_ROOT, "eval"),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
    )

    # checkpoint 回调
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CKPT_DIR,
        name_prefix="ckpt_tqc_her",
        save_replay_buffer=True,
        save_vecnormalize=False,
    )

    print(f"开始训练 TQC + HER on {ENV_ID}")
    start = time.time()

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    end = time.time()
    print(f"训练完成，总步数 {TOTAL_TIMESTEPS}，耗时 {end - start:.1f} 秒")

    final_model_path = os.path.join(LOG_ROOT, "tqc_her_pnp")
    model.save(final_model_path)
    print(f"最终模型已保存到: {final_model_path}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
