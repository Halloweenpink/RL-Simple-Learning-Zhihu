import os
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

ENV_ID = "Pusher-v5"

MODEL_PATH = "sac_pusher_n6"
NEW_MODEL_PATH = "sac_pusher_n7"
# 1:37.1 3:-31.7 4:-31.9
# n 1:35.2 2:30
def make_env():
    return gym.make(ENV_ID)  # 默认训练环境
    # return gym.make(ENV_ID, reward_near_weight=0.2, reward_dist_weight=5, reward_control_weight=0.01)  # 课程学习，研究后期改变奖励函数权重的影响

# 并行 16 个环境采样，加快采样速度（SAC 也支持 VecEnv）
train_env = make_vec_env(make_env, n_envs=16)

# ---- 如果有旧模型，就加载继续训；否则新建一个 ----
if os.path.exists(MODEL_PATH + ".zip"):
    print(f"加载已有 SAC 模型：{MODEL_PATH}.zip，继续训练...")
    model = SAC.load(MODEL_PATH, env=train_env, device="cuda")  # 如果没有 GPU 就改成 "cpu"
    model.learning_starts = 100_000
else:
    print("未找到旧 SAC 模型，重新初始化一个新模型...")
    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        learning_starts=10_000,
        # ---- 为了“训得快一点”的超参设置 ----
        learning_rate=1e-3,     # 比默认略大一点，加快收敛（如果不稳可以改回 3e-4）
        buffer_size=1_000_000,    # 不用 1e6 那么大，warm-up 更短一点
        batch_size=256,         # 利用 GPU，一次训练多点样本
        gamma=0.99,
        tau=0.005,              # 目标网络软更新系数
        train_freq=1,           # 每 step 都更新（有 16 个 env，每步收 16 个 transition）
        gradient_steps=1,       # 每次更新一步；如果你想“学得更猛”可以改成 2 或 4
        ent_coef="auto",        # 自动调节熵系数，省心
        device="cuda",          # 有 GPU 就用 "cuda"，没有就用 "cpu"
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256, 256],
                qf=[256, 256, 256],
            )
        )

    )

# ---- 继续训练（或者第一次训练）----
# reset_num_timesteps=False 表示「接着上次的步数训练」，而不是从 0 重新计数（sb3 nb）
model.learn(
    total_timesteps=3_000_000,
    reset_num_timesteps=False
)

# 训练完保存一下
model.save(NEW_MODEL_PATH)
train_env.close()
