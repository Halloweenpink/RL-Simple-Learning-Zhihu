import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3 import SAC
import numpy as np

test_env = gym.make("Pusher-v5", render_mode="human")
# test_env = gym.make("Pusher-v5", reward_near_weight=0, reward_dist_weight=1, reward_control_weight=0)
# model = PPO.load("sac_pusher_n5")
model = SAC.load("sac_pusher_n6",device="cuda")

obs, info = test_env.reset(seed=42)
avg = []
for i in range(50_000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    # print(str(i+1)+str(reward))
    if (i+1) % 100 == 0:
        # print("---------------------------------------")
        avg.append(reward)

    time.sleep(0.05)
    if terminated or truncated:
        obs, info = test_env.reset()
        time.sleep(0.4)

test_env.close()
print("!" + str(np.average(avg)))


"""sac avg object pos reward:
1:-24.09 2:-20.64 3:-19.60 4: -19.04
n:-22.97 n2:-19.79 n3：-17.994(调了weight) n4: -19.38（调了weight） n5: -16.63

sac ultimate obj pos rew:
4:-0.10784(2m), 
n:-0.19816, n2:-0.1298, n3:-0.08788, n4:-0.13029, n5:-0.0781，n6:-0.064318(3m)
下次记得画图打出来，做数据分析还是得有对应合适方法"""
