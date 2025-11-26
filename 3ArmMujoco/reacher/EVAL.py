import time
import gymnasium as gym
from stable_baselines3 import PPO


test_env = gym.make("Reacher-v5", render_mode="human")
model = PPO.load("ppo_reacher4")

obs, info = test_env.reset(seed=42)
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    time.sleep(0.1)
    if terminated or truncated:
        obs, info = test_env.reset()
        time.sleep(0.8)

test_env.close()
