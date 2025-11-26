import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

ENV_ID = "Pusher-v5"
MODEL_PATH = "ppo_pusher"   # SB3 实际保存的是 ppo_reacher.zip
NEW_MODEL_PATH = "ppo_pusher"   # SB3 实际保存的是 ppo_reacher.zip
def make_env():
    return gym.make(ENV_ID)

# 并行 8 个环境训练
train_env = make_vec_env(make_env, n_envs=16)

# ---- 关键：如果有旧模型，就加载继续训；否则新建一个 ----
if os.path.exists(MODEL_PATH + ".zip"):
    print(f"加载已有模型：{MODEL_PATH}.zip，继续训练...")
    model = PPO.load(MODEL_PATH, env=train_env)
    # 保险起见再 set_env 一次（有些版本需要）
    model.set_env(train_env)
else:
    print("未找到旧模型，重新初始化一个新模型...")
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        learning_rate=1e-3,
    )

# ---- 继续训练（或者第一次训练）----
# 注意：reset_num_timesteps=False 表示「接着上次的步数训练」，而不是从 0 重新计数 sb3内置了，nb啊sb3
model.learn(
    total_timesteps=500_000,
    reset_num_timesteps=False
)

# 训练完保存一下
model.save(NEW_MODEL_PATH)
train_env.close()
