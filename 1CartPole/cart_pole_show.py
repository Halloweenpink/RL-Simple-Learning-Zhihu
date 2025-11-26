import gymnasium as gym
import os
# os.environ["SDL_VIDEODRIVER"] = "x11"  # 👉 强制 SDL（pygame 的底层图形库） 使用 X11 驱动来创建窗口，否则它可能选到 dummy（虚拟显示）或 wayland 导致黑屏。
# os.environ["SDL_RENDER_DRIVER"] = "software"  # 👉 告诉 SDL 用 软件渲染（CPU 绘制像素），而不是 OpenGL（GPU）或硬件加速。

# Initialise the environment
print("env created")
env = gym.make("LunarLander-v3", render_mode="human")
# env = gym.make("LunarLander-v3", render_mode=None)

print("env made, resetting…")

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
for _ in range(2000):
    # this is where you would insert your policy
    action = env.action_space.sample()

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

    print(_)

env.close()
