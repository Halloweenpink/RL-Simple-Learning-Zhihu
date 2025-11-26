import time

from sb3_contrib import TQC
import gymnasium as gym
import gymnasium_robotics

env = gym.make("FetchPickAndPlace-v4", render_mode="human")
model = TQC.load("../../3ArmMujoco/欺骗我跑出来的HuggingFace SAC超参/ckpt_tqc_her_50000_steps.zip", env=env)

# 1m有时候可以把物体移到平面；1.5m经常把方块踢下去、但是可以夹到高点了；2m会晃；2.5m运到位置之后要么远离、要么不动，但是有时候会把方块踢下去；
# 3m意识到假如停了乱动会降低reward，同时平面的移动会夹起来送过去不是踢过去。
num_episodes = 10
for ep in range(num_episodes):
    obs, info = env.reset()
    done = False
    truncated = False
    episode_reward = 0
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        episode_reward += reward
        time.sleep(0.07)
    print(f"Episode {ep+1} reward: {episode_reward}")
env.close()
