import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

ENV_ID = "Reacher-v5"

def make_env():
    return gym.make(ENV_ID)

# 并行 8 个环境训练
train_env = make_vec_env(make_env, n_envs=8)

model = PPO(
    policy="MlpPolicy",
    env=train_env,
    verbose=1,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    learning_rate=3e-4,
)

model.learn(total_timesteps=300_000)
model.save("ppo_reacher")
train_env.close()
