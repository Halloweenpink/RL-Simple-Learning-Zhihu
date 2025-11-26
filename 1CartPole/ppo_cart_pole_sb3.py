""" CartPole 是用常微分方程模型的环境，是一个玩具环境。很好训练收敛。"""

# # 每一个训练都渲染

# import gymnasium as gym
# from stable_baselines3 import PPO

# env = gym.make("CartPole-v1", render_mode="human")
# model = PPO("MlpPolicy", env, verbose=1, device="cpu") # 或 device="cuda"
# model.learn(total_timesteps=100_000, progress_bar=True)

# obs, _ = env.reset()
# for _ in range(10):
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, done, truncated, info = env.step(action)
#     env.render()
#     if done or truncated:
#         obs, _ = env.reset()



# # worker并行 + 训练完了渲染
# import gymnasium as gym
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
#
# train_env = make_vec_env("CartPole-v1", n_envs=8)
# model = PPO("MlpPolicy", train_env, verbose=1, device="cpu")
# model.learn(total_timesteps=200_000, progress_bar=True)
# train_env.close()
#
# test_env = gym.make("CartPole-v1", render_mode="human")
# obs, _ = test_env.reset()
# for _ in range(1000):
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = test_env.step(action)
#     if terminated or truncated:
#         obs, _ = test_env.reset()
# test_env.close()


# 带tensorboard版

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

# 1) 训练环境（向量化）。make_vec_env 会自动加 Monitor 包装器
train_env = make_vec_env("CartPole-v1", n_envs=8)

# 2) 开启 TensorBoard：tensorboard_log 指定目录
model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    device="cpu",
    tensorboard_log="./tb_logs/cartpole"  # <— 关键
)

# 3) 评估环境（单环境 + Monitor，便于 eval/ 指标写入 TensorBoard）
eval_env = Monitor(gym.make("CartPole-v1"))

# 4) 评估回调：定期评估并把 mean_reward 等写入 eval/ 名字空间；同时保存最好模型
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/cartpole_best",
    log_path="./tb_logs/cartpole",
    eval_freq=10_000,             # 每隔多少步评估一次
    n_eval_episodes=10,           # 每次评估跑多少回合
    deterministic=True,
    render=False
)

# 5) 训练并指定 tb_log_name 作为这次 run 的名称（TensorBoard 曲线分组名）
model.learn(
    total_timesteps=200_000,
    progress_bar=True,
    tb_log_name="ppo_run_1"       # <— 关键
    , callback=eval_callback
)

train_env.close()

# 6) 测试渲染（与 TensorBoard 无关）
test_env = gym.make("CartPole-v1", render_mode="human")
obs, _ = test_env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    if terminated or truncated:
        obs, _ = test_env.reset()
test_env.close()
