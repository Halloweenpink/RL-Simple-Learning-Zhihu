import gymnasium as gym
import gymnasium_robotics  # 有了这行，才能调用fetch包

gym.register_envs(gymnasium_robotics)

# render_mode="human" 会弹一个窗口，看到机械臂和方块
env = gym.make("FetchPickAndPlace-v4", render_mode="human")

obs, info = env.reset()
print("Observation keys:", obs.keys())
print("Initial obs['observation'].shape:", obs["observation"].shape)

for step in range(2000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("Episode ended. Reward:", reward)
        obs, info = env.reset()

env.close()

