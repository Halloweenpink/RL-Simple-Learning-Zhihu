"""Box2D是一个2D刚体动力学模拟器，用于模拟2D物理环境。
LunarLanderContinuous 2D 连续推力 单体刚体 姿态 + 碰撞着陆 有地面碰撞"""

import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# --- 1) 环境与超参 ---
ENV_ID = "LunarLanderContinuous-v3"
SEED = 0

# # 训练用向量并行环境（更稳更快）
# train_env = make_vec_env(ENV_ID, n_envs=8, seed=SEED)  # 默认创建的是同步的向量环境

# # --- 2) 建PPO 在连续动作空间中默认使用 高斯策略，策略网络输出的是动作分布的「均值 μ 和对数方差 log σ」，可微分的正态分布 ---
# model = PPO(
#     policy="MlpPolicy",
#     env=train_env,
#     verbose=1,
#     seed=SEED,
#     device="cpu",          # sb3对这个问题用cpu写的，cpu更快
# )

# # --- 3) 训练 ---
# model.learn(total_timesteps=250_000)  # 21万次ep_rew才34
# model.save("./ppo_lunarlander_cont")  # continuous
# train_env.close()

# # --- 4) 评估（无渲染）---
# eval_env = gym.make(ENV_ID)
# mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
# print(f"Eval over 10 eps: mean_reward={mean_reward:.1f} +/- {std_reward:.1f}")
# eval_env.close()
# # 20w:Eval over 10 eps: mean_reward=201.6 +/- 77.7

# --- 5) 演示一局（带可视化）---
# 注意：渲染要新建 render_mode="human" 的环境；VecEnv 不支持可视化
loaded_model = PPO.load(os.path.abspath("ppo_lunarlander_cont"), device="cpu") # 使用绝对路径

demo_env = gym.make(ENV_ID, render_mode="human")
obs, info = demo_env.reset(seed=SEED)

for _ in range(2000):
    action, _ = loaded_model.predict(obs, deterministic=True)
    print(action)
    obs, reward, terminated, truncated, info = demo_env.step(action)
    if terminated or truncated:
        obs, _ = demo_env.reset()
demo_env.close()
