import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np

env = gym.make("Reacher-v5", render_mode="human")
obs, info = env.reset(seed=0)

# 随机跑一跑
for t in range(200):
    action = env.action_space.sample()  # 动作: Box(-1,1,(2,))
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()

# mjviewer
mj_model = env.unwrapped.model
mj_data = env.unwrapped.data

with mujoco.viewer.launch(mj_model, mj_data) as viewer:
    for t in range(2000):
        # 你可以用随机策略，也可以用训练好的策略
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # 这里不用再 mj_step，因为 env.step 里已经调用过 mujoco.mj_step 了
        viewer.sync()

        if terminated or truncated:
            obs, info = env.reset()