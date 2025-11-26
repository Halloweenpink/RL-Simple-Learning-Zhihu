# ppo_cartpole_vecenv_tb.py
import os, time, numpy as np, torch, torch.nn as nn, torch.optim as optim
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from dataclasses import dataclass
from typing import Tuple
from torch.utils.tensorboard import SummaryWriter

@dataclass # 装饰器，用于自动生成类的特殊方法， __init__、__repr__ 等
class PPOConfig:
    env_id: str = "CartPole-v1"
    seed: int = 42
    total_timesteps: int = 800_000

    # 采样并行
    n_envs: int = 16
    rollout_len: int = 1024  # 每个环境每轮收集步数（总 batch = n_envs * rollout_len）

    # PPO 更新
    n_epochs: int = 10
    minibatch_size: int = 4096  # 按 (T*N) 计 T是rollout_len，N是n_envs
    gamma: float = 0.99  # GAEl里面的折扣因子，用于计算未来奖励的现值 
    gae_lambda: float = 0.95  # GAE 里的 Lambda 参数，用于计算优势函数
    clip_coef: float = 0.2  # PPO 里的 epsilon 参数，用于限制策略更新的范围 clamp(ratio, 1-ε, 1+ε)
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5  # 见函数nn.utils.clip_grad_norm_，用于限制梯度范数，防止梯度爆炸
    target_kl: float = 0.03

    device: str = "cuda"   # 可改 "cuda"
    log_dir: str = "runs/ppo_cartpole"
    use_async: bool = False  # Windows 上 SyncVectorEnv 往往更稳

cfg = PPOConfig() # 使用了无参构造，意味着使用了类中定义的所有默认参数值

def make_env(env_id, seed_offset=0):
    def thunk():  # 延迟计算函数
        env = gym.make(env_id)  # 这是一个闭包：捕获并保存外部参数
        # 兼容不同 Gym/Gymnasium 版本的 RecordEpisodeStatistics 包装器
        try:
            env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1000)  # gym的内置环境包装器，用于记录每个episode的统计信息
        except TypeError:
            env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=cfg.seed + seed_offset)
        return env
    return thunk

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),  # tanh用于进入非线性映射，不用softmax因为tanh的性质更适合
            nn.Linear(64, 64), nn.Tanh(),
        )
        self.pi_head = nn.Linear(64, act_dim)
        self.v_head  = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.feature(x)
        return self.pi_head(h), self.v_head(h).squeeze(-1)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor = None):
        logits, value = self.forward(obs) # 输出层的输出在神经网络里面都叫logits，因为最开始的历史过了softmax激活
        dist = torch.distributions.Categorical(logits=logits) # 分类分布适用于离散动作空间；Categorical分布内部计算时使用logits可以避免重复计算softmax
        if action is None:
            action = dist.sample()
        logp = dist.log_prob(action)
        ent = dist.entropy()
        return action, logp, ent, value

class VecRolloutBuffer:
    """按 (T, N) 存，GAE 沿时间维计算；更新时展平为 (T*N, …)。"""
    def __init__(self, T: int, N: int, obs_dim: int, device: str):
        self.T, self.N = T, N
        self.obs = torch.zeros((T, N, obs_dim), dtype=torch.float32, device=device)  # 存储观测值s
        self.actions = torch.zeros((T, N), dtype=torch.long, device=device)
        self.logprobs = torch.zeros((T, N), dtype=torch.float32, device=device)  # 存储动作a的log概率logπ(a|s)
        self.rewards = torch.zeros((T, N), dtype=torch.float32, device=device)
        self.dones = torch.zeros((T, N), dtype=torch.float32, device=device)  # 存储环境是否结束的标志
        self.values = torch.zeros((T, N), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((T, N), dtype=torch.float32, device=device)
        self.returns = torch.zeros((T, N), dtype=torch.float32, device=device)

    def compute_gae(self, last_values: torch.Tensor, gamma: float, lam: float):
        T, N = self.T, self.N
        adv = torch.zeros((N,), dtype=torch.float32, device=last_values.device)
        for t in reversed(range(T)):
            next_value = last_values if t == T - 1 else self.values[t + 1]
            next_nonterminal = 1.0 - self.dones[t]  # (N,) nonterminal是环境是否结束的标志，和done的区别是done是环境是否结束，nonterminal是环境是否继续运行
            delta = self.rewards[t] + gamma * next_value * next_nonterminal - self.values[t]
            adv = delta + gamma * lam * next_nonterminal * adv
            self.advantages[t] = adv
        self.returns = self.advantages + self.values

    def minibatches(self, batch_size: int):
        obs = self.obs.reshape(-1, self.obs.shape[-1])
        actions = self.actions.reshape(-1)
        logprobs = self.logprobs.reshape(-1)
        values = self.values.reshape(-1)
        returns = self.returns.reshape(-1)
        adv = self.advantages.reshape(-1)
        # 标准化优势
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        idx = np.random.permutation(obs.shape[0])  # idx是一个随机 permutation 数组，用于随机采样 mini-batch，permute是数学上的随机排列
        for start in range(0, idx.shape[0], batch_size):
            mb = idx[start:start + batch_size]
            yield obs[mb], actions[mb], logprobs[mb], values[mb], returns[mb], adv[mb]

def evaluate(env_id, net, episodes=10, seed=123, device="cpu"):
    env = gym.make(env_id)
    rs, lens, succ = [], [], []  # rs是奖励列表，lens是步数列表，succ是是否成功列表
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, ep_r, ep_l, ok = False, 0.0, 0, False  # ep_r是奖励，ep_l是步数，ok是是否成功
        while not done:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits, _ = net.forward(obs_t)
                action = logits.argmax(dim=-1).item()  # 确定性策略！！！
            obs, r, term, trunc, info = env.step(action)
            ep_r += r; ep_l += 1
            done = term or trunc
            if done and trunc:
                ok = True
        rs.append(ep_r); lens.append(ep_l); succ.append(ok)
    env.close()
    return float(np.mean(rs)), float(np.std(rs)), float(np.mean(lens)), float(np.mean(succ))

def train(cfg: PPOConfig):
    set_seed(cfg.seed)
    torch.set_num_threads(max(1, os.cpu_count() // 2))  # cpu_count()是获取当前系统的CPU核心数，//2是为了避免占用所有核心，max(1, ...)是为了避免线程数为0

    env_fns = [make_env(cfg.env_id, i) for i in range(cfg.n_envs)]
    Vec = AsyncVectorEnv if cfg.use_async else SyncVectorEnv  
    ''' - AsyncVectorEnv （异步向量化环境）：
    - 使用Python的多进程并行运行多个环境实例
    - 每个环境在独立进程中运行，真正并行处理
    - 适合计算密集型环境，能充分利用多核CPU
    - 注意：Windows上多进程可能存在稳定性问题

    - SyncVectorEnv （同步向量化环境）：
    - 在单个进程中顺序运行多个环境
    - 通过循环依次与每个环境交互
    - 计算效率低于异步方式，但在Windows上通常更稳定'''
    env = Vec(env_fns)

    obs, infos = env.reset(seed=cfg.seed)  # 重置环境，返回初始观测值和环境信息！！
    obs_dim = obs.shape[1]
    act_dim = env.single_action_space.n

    device = torch.device(cfg.device)
    net = ActorCritic(obs_dim, act_dim).to(device)
    optim_ = optim.Adam(net.parameters(), lr=cfg.learning_rate,amsgrad=True)

    from pathlib import Path  # >>> 新增
    # >>> 日志：使用“绝对路径 + 时间戳子目录”，并打印出来
    runs_root = Path(r"C:\Users\H16\PycharmProjects\PythonProject1\runs")
    run_name = time.strftime("ppo_cartpole-%Y%m%d-%H%M%S") # 时间戳格式化字符串，%Y%m%d-%H%M%S是年-月-日-时-分-秒
    log_dir = runs_root / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir), purge_step=0)
    print("[TB] writing to:", log_dir.resolve())



    start_time = time.time()
    global_steps = 0 # 全局步数，记录训练了多少步
    update_idx = 0

    while global_steps < cfg.total_timesteps:
        buf = VecRolloutBuffer(cfg.rollout_len, cfg.n_envs, obs_dim, device)

        # ======= 采样 =======
        ep_cnt = 0  # 用于日志，cnt的全称是count，记录采样了多少个回合
        for t in range(cfg.rollout_len):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                act, logp, ent, val = net.get_action_and_value(obs_t)
            act_np = act.cpu().numpy()
            next_obs, reward, terminated, truncated, infos = env.step(act_np)
            done = np.logical_or(terminated, truncated).astype(np.float32)

            # 存
            buf.obs[t] = obs_t
            buf.actions[t] = act
            buf.logprobs[t] = logp
            buf.values[t] = val
            buf.rewards[t] = torch.tensor(reward, dtype=torch.float32, device=device)
            buf.dones[t] = torch.tensor(done, dtype=torch.float32, device=device)

            # 从 final_info 里读取回合统计并写 TensorBoard
            if "final_info" in infos:
                for fin in infos["final_info"]:
                    if fin is not None and "episode" in fin:
                        ep = fin["episode"]  # {'r': return, 'l': length, 't': elapsed_time}
                        writer.add_scalar("charts/episode_return", ep["r"], global_steps)
                        writer.add_scalar("charts/episode_length", ep["l"], global_steps)
                        ep_cnt += 1

            obs = next_obs
            global_steps += cfg.n_envs

        # ======= GAE/Returns =======
        with torch.no_grad(): # PyTorch默认会为所有张量计算梯度！！no_grad()是为了避免计算梯度，因为在计算GAE时不需要梯度
            last_v = net.get_action_and_value(
                torch.tensor(obs, dtype=torch.float32, device=device)
            )[3]  # (N,)
        buf.compute_gae(last_v, cfg.gamma, cfg.gae_lambda)

        # ======= PPO 更新 =======
        update_idx += 1
        pg_loss_log = v_loss_log = ent_log = kl_log = clip_frac_log = 0.0  # 初始化
        sgd_steps = 0  # sgd的全称是stochastic gradient descent

        for _ in range(cfg.n_epochs):
            for mb_obs, mb_act, mb_logp_old, mb_v_old, mb_ret, mb_adv in buf.minibatches(cfg.minibatch_size):
                _, mb_logp, mb_ent, mb_v = net.get_action_and_value(mb_obs, mb_act)
                logratio = mb_logp - mb_logp_old  # 当前策略（更新后的网络）和旧策略（采样时的网络）的对数概率之差
                ratio = torch.exp(logratio)  # 通过指数操作，我们将对数空间的差值转换回概率空间的比率

                # policy loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef) # clamp操作，限制在一个范围内
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                clip_frac = (torch.abs(ratio - 1.0) > cfg.clip_coef).float().mean()

                # value loss
                v_loss = 0.5 * (mb_ret - mb_v).pow(2).mean()

                # entropy
                ent_loss = -mb_ent.mean()

                loss = pg_loss + cfg.vf_coef * v_loss + cfg.ent_coef * ent_loss
                optim_.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm) 
                    # - 优化器层面的约束，通过限制参数梯度的L2范数来防止梯度爆炸
                    # - 这种方法与具体算法无关，是深度学习中通用的稳定训练技术
                optim_.step()  # 更新网络参数！！！虽然没有显示输入梯度，但是pytorch自动保存在里面了

                with torch.no_grad():
                    approx_kl = (ratio - 1 - logratio).mean().abs().item()

                # 聚合日志
                pg_loss_log += pg_loss.item() # item()将张量转换为Python标量
                v_loss_log += v_loss.item()
                ent_log += (-ent_loss).item()
                kl_log += approx_kl
                clip_frac_log += clip_frac.item()
                sgd_steps += 1

            if sgd_steps and (kl_log / sgd_steps) > cfg.target_kl:
                break

        # ======= 写入训练过程标量 =======
        if sgd_steps:
            writer.add_scalar("loss/policy", pg_loss_log / sgd_steps, global_steps)
            writer.add_scalar("loss/value", v_loss_log / sgd_steps, global_steps)
            writer.add_scalar("loss/entropy", ent_log / sgd_steps, global_steps)
            writer.add_scalar("diagnostics/approx_kl", kl_log / sgd_steps, global_steps)
            writer.add_scalar("diagnostics/clip_frac", clip_frac_log / sgd_steps, global_steps)
            writer.add_scalar("diagnostics/learning_rate", optim_.param_groups[0]["lr"], global_steps) # param_groups是参数组，每个组对应一个优化器，但是我们这里只有一个
            writer.add_scalar("charts/fps", global_steps / (time.time() - start_time + 1e-8), global_steps)
            writer.add_scalar("charts/episodes_in_update", ep_cnt, global_steps)

        # ======= 可选：定期做确定性评估 =======
        if update_idx % 5 == 0:
            mean_r, std_r, mean_len, succ = evaluate(cfg.env_id, net, episodes=20, device=cfg.device)
            writer.add_scalar("eval/return_mean", mean_r, global_steps)
            writer.add_scalar("eval/return_std", std_r, global_steps)
            writer.add_scalar("eval/ep_len_mean", mean_len, global_steps)
            writer.add_scalar("eval/timeLimit_success_rate", succ, global_steps)
            print(f"[EVAL] step={global_steps}  R={mean_r:.1f}±{std_r:.1f}  len={mean_len:.1f}")

    env.close()
    writer.close()
    return net

def render_policy(net: ActorCritic, cfg: PPOConfig, episodes: int = 20):
    env = gym.make(cfg.env_id, render_mode="human")
    obs, _ = env.reset(seed=cfg.seed + 999) 
    device = torch.device(cfg.device)
    for ep in range(episodes):
        done = False; ret = 0.0
        while not done:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                a, _, _, _ = net.get_action_and_value(obs_t)
            obs, r, term, trunc, _ = env.step(a.item())
            ret += r; done = term or trunc
        print(f"[Replay] ep_return={ret:.1f}")
        obs, _ = env.reset()
    env.close()

if __name__ == "__main__":
    net = train(cfg)
    render_policy(net, cfg)
