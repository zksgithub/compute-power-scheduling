"""
算力 - 电力协同优化 - PPO 深度强化学习实现

基于 PyTorch 实现 PPO 算法，用于算力 - 电力协同调度优化
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import json
from datetime import datetime
import csv


# ==================== 1. 环境定义 ====================

class ComputePowerEnv:
    """算力 - 电力协同优化环境"""
    
    def __init__(self, data_path, config=None):
        self.data_path = data_path
        self.config = config or {}
        
        # 加载数据
        self.load_data()
        
        # 环境参数
        self.T = 24  # 24 小时
        self.action_dim = 3  # [充电功率，放电功率，负载转移比例]
        
        # 储能参数
        self.storage = {
            'capacity': 100,  # MWh
            'max_charge': 20,  # MW
            'max_discharge': 20,  # MW
            'efficiency_ch': 0.95,
            'efficiency_dis': 0.95,
            'min_soc': 10,  # 最小 SOC (%)
            'max_soc': 90   # 最大 SOC (%)
        }
        
        # 电价（元/kWh）
        self.electricity_price = {
            'peak': 1.2,
            'flat': 0.8,
            'valley': 0.4
        }
        
        # 碳强度（kgCO₂/kWh）
        self.carbon_intensity = 0.58
        
        # 重置环境
        self.reset()
    
    def load_data(self):
        """加载数据集"""
        # 加载电力数据
        power_data = []
        try:
            with open(f'{self.data_path}/shandong_power_classified.csv', 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    power_data.append(row)
        except:
            power_data = []
        
        # 统计容量
        capacity_by_type = {}
        for row in power_data:
            etype = row.get('能源类别', 'unknown')
            cap_str = row.get('Capacity (MW)', '0')
            try:
                cap = float(cap_str) if cap_str else 0
            except:
                cap = 0
            capacity_by_type[etype] = capacity_by_type.get(etype, 0) + cap
        
        self.renewable_capacity = capacity_by_type.get('新能源', 132379.1)  # MW
        
        # 加载 GPU 数据
        gpu_data = []
        try:
            with open(f'{self.data_path}/node_info_df.csv', 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    gpu_data.append(row)
        except:
            gpu_data = []
        
        self.total_gpus = 0
        for row in gpu_data:
            cap_str = row.get('gpu_capacity_num', '0')
            try:
                cap = int(cap_str) if cap_str else 0
                self.total_gpus += cap
            except:
                pass
        
        if self.total_gpus == 0:
            self.total_gpus = 10412  # 默认值
        
        # 加载任务数据
        task_data = []
        try:
            with open(f'{self.data_path}/data_trace_processed.csv', 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    task_data.append(row)
        except:
            task_data = []
        
        exec_times = []
        for row in task_data:
            exec_str = row.get('exec_time_seconds', '0')
            try:
                exec_time = float(exec_str) if exec_str else 0
                if exec_time > 0:
                    exec_times.append(exec_time)
            except:
                pass
        
        self.avg_exec_time = np.mean(exec_times) if exec_times else 27.2
    
    def get_tou(self, hour):
        """获取时段类型"""
        if hour in range(8, 11) or hour in range(17, 22):
            return 'peak'
        elif hour in range(11, 17):
            return 'flat'
        else:
            return 'valley'
    
    def generate_scenario(self, seed=42):
        """生成 24 小时场景"""
        np.random.seed(seed)
        
        hours = list(range(24))
        
        # 新能源出力
        renewable = []
        for h in hours:
            wind_factor = 0.6 + 0.4 * np.sin(2 * np.pi * (h + 6) / 24)
            if 6 <= h <= 18:
                solar_factor = np.sin(np.pi * (h - 6) / 12)
            else:
                solar_factor = 0
            
            wind_factor *= (0.8 + 0.4 * np.random.random())
            solar_factor *= (0.8 + 0.4 * np.random.random())
            
            wind_output = self.renewable_capacity * 0.004 * wind_factor
            solar_output = self.renewable_capacity * 0.003 * solar_factor
            
            renewable.append(wind_output + solar_output)
        
        # 负载
        load = []
        for h in hours:
            base_load = self.total_gpus * 55 / 1000 * 0.5  # MW
            
            if 9 <= h <= 18:
                hour_factor = 1.5
            elif 19 <= h <= 22:
                hour_factor = 1.2
            else:
                hour_factor = 0.8
            
            random_factor = 0.9 + 0.2 * np.random.random()
            load.append(base_load * hour_factor * random_factor)
        
        return np.array(renewable), np.array(load)
    
    def reset(self, seed=42):
        """重置环境"""
        self.renewable, self.load = self.generate_scenario(seed)
        self.soc = 50.0  # 初始 SOC 50%
        self.t = 0
        self.total_cost = 0
        self.total_carbon = 0
        self.total_renewable_used = 0
        
        return self._get_state()
    
    def _get_state(self):
        """获取状态"""
        # 状态：[小时，新能源，负载，SOC，电价，碳强度]
        hour = self.t
        renew = self.renewable[self.t] / 600  # 归一化
        load = self.load[self.t] / 500  # 归一化
        soc = self.soc / 100
        tou = self.get_tou(hour)
        price = self.electricity_price[tou] / 1.2  # 归一化
        carbon = self.carbon_intensity / 0.85  # 归一化
        
        state = np.array([hour / 24, renew, load, soc, price, carbon], dtype=np.float32)
        return state
    
    def step(self, action):
        """执行动作"""
        # 动作：[充电功率，放电功率，负载转移比例]
        # 动作范围：[-1, 1]，需要转换为实际值
        charge_power = (action[0] + 1) / 2 * self.storage['max_charge']  # 0-20 MW
        discharge_power = (action[1] + 1) / 2 * self.storage['max_discharge']  # 0-20 MW
        load_shift = action[2]  # -1 到 1
        
        # 互斥约束：不能同时充放电
        if charge_power > 0.1 and discharge_power > 0.1:
            if charge_power > discharge_power:
                discharge_power = 0
            else:
                charge_power = 0
        
        # SOC 约束
        if self.soc >= self.storage['max_soc']:
            charge_power = 0
        if self.soc <= self.storage['min_soc']:
            discharge_power = 0
        
        # 更新 SOC
        if charge_power > 0:
            self.soc += charge_power * self.storage['efficiency_ch'] / self.storage['capacity'] * 100
        if discharge_power > 0:
            self.soc -= discharge_power / self.storage['efficiency_dis'] / self.storage['capacity'] * 100
        
        self.soc = np.clip(self.soc, self.storage['min_soc'], self.storage['max_soc'])
        
        # 计算电网购电
        renew_available = self.renewable[self.t]
        load_demand = self.load[self.t]
        
        # 负载转移（将部分负载转移到低谷时段）
        if load_shift > 0:
            shifted_load = load_demand * load_shift * 0.1  # 最多转移 10%
            load_demand -= shifted_load
        else:
            shifted_load = 0
        
        # 新能源消纳
        renewable_used = min(renew_available, load_demand + charge_power)
        self.total_renewable_used += renewable_used
        
        # 电网购电
        grid_power = max(0, load_demand - renew_available - discharge_power + charge_power)
        
        # 计算成本和碳排放
        tou = self.get_tou(self.t)
        price = self.electricity_price[tou]
        
        cost = grid_power * price
        carbon = grid_power * self.carbon_intensity
        
        self.total_cost += cost
        self.total_carbon += carbon
        
        # 计算奖励
        # 奖励设计：成本 + 碳排放 - 新能源消纳
        reward = -(cost * 1.0 + carbon * 3.0)  # 增加碳排放权重
        reward -= abs(load_shift) * 0.05  # 负载转移惩罚
        reward -= abs(charge_power - discharge_power) * 0.005  # 储能动作平滑惩罚
        
        # 额外奖励：使用新能源
        reward += renewable_used * 0.01
        
        # 更新时段
        self.t += 1
        done = (self.t >= self.T)
        
        if done:
            renewable_rate = self.total_renewable_used / sum(self.renewable) * 100
            reward += renewable_rate * 0.1  # 消纳率奖励
        
        next_state = self._get_state() if not done else np.zeros(6, dtype=np.float32)
        
        info = {
            'cost': cost,
            'carbon': carbon,
            'renewable_used': renewable_used,
            'soc': self.soc
        }
        
        return next_state, reward, done, info


# ==================== 2. PPO 网络定义 ====================

class ActorCritic(nn.Module):
    """Actor-Critic 网络"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        # Actor 网络
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 输出范围 [-1, 1]
        )
        
        # Critic 网络
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 动作标准差（可学习参数）
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
    
    def forward(self, state):
        """前向传播"""
        mean = self.actor(state)
        value = self.critic(state)
        std = self.log_std.exp()
        return mean, value, std
    
    def get_action(self, state, deterministic=False):
        """获取动作"""
        mean, value, std = self.forward(state)
        
        if deterministic:
            action = mean
        else:
            dist = Normal(mean, std)
            action = dist.sample()
        
        return action, value
    
    def evaluate(self, state, action):
        """评估动作（用于 PPO 更新）"""
        mean, value, std = self.forward(state)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=1, keepdim=True)
        entropy = dist.entropy().sum(dim=1, keepdim=True)
        return log_prob, value, entropy


# ==================== 3. PPO 训练器 ====================

class PPOTrainer:
    """PPO 训练器"""
    
    def __init__(self, env, config=None):
        self.env = env
        self.config = config or {}
        
        # 超参数
        self.gamma = self.config.get('gamma', 0.99)
        self.gae_lambda = self.config.get('gae_lambda', 0.95)
        self.clip_epsilon = self.config.get('clip_epsilon', 0.2)
        self.lr = self.config.get('lr', 3e-4)
        self.epochs = self.config.get('epochs', 10)
        self.batch_size = self.config.get('batch_size', 64)
        self.value_coef = self.config.get('value_coef', 0.5)
        self.entropy_coef = self.config.get('entropy_coef', 0.01)
        
        # 网络
        state_dim = 6
        action_dim = 3
        hidden_dim = self.config.get('hidden_dim', 128)
        
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        
        # 训练统计
        self.episode_rewards = []
        self.episode_costs = []
        self.episode_carbons = []
        self.episode_renewable_rates = []
    
    def select_action(self, state, deterministic=False):
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.policy.get_action(state_tensor, deterministic)
        return action.squeeze(0).numpy()
    
    def compute_gae(self, rewards, values, dones):
        """计算 Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages).unsqueeze(1)
        returns = advantages + torch.FloatTensor(values).unsqueeze(1)
        
        return advantages, returns
    
    def update(self, states, actions, rewards, values, log_probs, dones):
        """PPO 更新"""
        advantages, returns = self.compute_gae(rewards, values, dones)
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 打乱数据
        dataset_size = len(states)
        indices = np.random.permutation(dataset_size)
        
        for epoch in range(self.epochs):
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = torch.FloatTensor(np.array(states)[batch_indices])
                batch_actions = torch.FloatTensor(np.array(actions)[batch_indices])
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = torch.FloatTensor(np.array(log_probs)[batch_indices])
                
                # 评估当前策略
                new_log_probs, new_values, entropies = self.policy.evaluate(batch_states, batch_actions)
                
                # 计算比率
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO 损失
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = nn.MSELoss()(new_values, batch_returns)
                
                # 熵损失
                entropy_loss = -entropies.mean()
                
                # 总损失
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # 优化
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
    
    def train(self, num_episodes=1000, save_interval=100):
        """训练"""
        print(f"开始训练 PPO，共 {num_episodes} 集...")
        
        for episode in range(num_episodes):
            state = self.env.reset(seed=episode)
            episode_reward = 0
            episode_cost = 0
            episode_carbon = 0
            
            states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
            
            for t in range(self.env.T):
                # 选择动作
                action, value = self.policy.get_action(torch.FloatTensor(state).unsqueeze(0))
                
                # 执行动作
                next_state, reward, done, info = self.env.step(action.squeeze(0).numpy())
                
                # 存储数据
                states.append(state)
                actions.append(action.squeeze(0).numpy())
                rewards.append(reward)
                values.append(value.squeeze(0).item())
                log_probs.append(self.policy.evaluate(torch.FloatTensor(state).unsqueeze(0), 
                                                       action)[0].squeeze(0).item())
                dones.append(done)
                
                state = next_state
                episode_reward += reward
                episode_cost += info['cost']
                episode_carbon += info['carbon']
            
            # 更新策略
            self.update(states, actions, rewards, values, log_probs, dones)
            
            # 统计
            renewable_rate = self.env.total_renewable_used / sum(self.env.renewable) * 100
            self.episode_rewards.append(episode_reward)
            self.episode_costs.append(episode_cost)
            self.episode_carbons.append(episode_carbon)
            self.episode_renewable_rates.append(renewable_rate)
            
            # 打印进度
            if (episode + 1) % save_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-save_interval:])
                avg_cost = np.mean(self.episode_costs[-save_interval:])
                avg_carbon = np.mean(self.episode_carbons[-save_interval:])
                avg_renew = np.mean(self.episode_renewable_rates[-save_interval:])
                
                print(f"Episode {episode+1}/{num_episodes}: "
                      f"Reward={avg_reward:.2f}, Cost={avg_cost:.2f}, "
                      f"Carbon={avg_carbon:.4f}, Renew={avg_renew:.1f}%")
        
        print("训练完成！")
        
        return {
            'rewards': self.episode_rewards,
            'costs': self.episode_costs,
            'carbons': self.episode_carbons,
            'renewable_rates': self.episode_renewable_rates
        }
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_costs': self.episode_costs,
            'episode_carbons': self.episode_carbons,
            'episode_renewable_rates': self.episode_renewable_rates
        }, path)
        print(f"模型已保存到 {path}")
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_costs = checkpoint.get('episode_costs', [])
        self.episode_carbons = checkpoint.get('episode_carbons', [])
        self.episode_renewable_rates = checkpoint.get('episode_renewable_rates', [])
        print(f"模型已从 {path} 加载")


# ==================== 4. 实验运行 ====================

def run_ppo_experiment(data_path, num_episodes=500):
    """运行 PPO 实验"""
    print("=" * 60)
    print("算力 - 电力协同优化 - PPO 深度强化学习实验")
    print("=" * 60)
    
    # 创建环境
    env = ComputePowerEnv(data_path)
    
    # 创建训练器
    config = {
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'lr': 5e-4,  # 增加学习率
        'epochs': 15,  # 增加更新轮次
        'batch_size': 32,  # 减小 batch size
        'hidden_dim': 256,  # 增加网络容量
        'value_coef': 0.5,
        'entropy_coef': 0.02  # 增加熵鼓励探索
    }
    
    trainer = PPOTrainer(env, config)
    
    # 训练（增加轮次）
    training_results = trainer.train(num_episodes=1000, save_interval=200)
    
    # 保存模型
    model_path = f'{data_path}/../ppo_model.pth'
    trainer.save(model_path)
    
    # 测试（确定性策略）
    print("\n测试训练好的模型...")
    state = env.reset(seed=123)
    test_cost = 0
    test_carbon = 0
    
    for t in range(env.T):
        action = trainer.select_action(state, deterministic=True)
        next_state, reward, done, info = env.step(action)
        test_cost += info['cost']
        test_carbon += info['carbon']
        state = next_state
    
    test_renewable_rate = env.total_renewable_used / sum(env.renewable) * 100
    
    print(f"测试结果：Cost={test_cost:.2f}, Carbon={test_carbon:.4f}, "
          f"Renewable={test_renewable_rate:.1f}%")
    
    # 保存实验结果
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'training': {
            'final_avg_cost': np.mean(training_results['costs'][-100:]),
            'final_avg_carbon': np.mean(training_results['carbons'][-100:]),
            'final_avg_renewable': np.mean(training_results['renewable_rates'][-100:])
        },
        'test': {
            'cost': test_cost,
            'carbon': test_carbon,
            'renewable_rate': test_renewable_rate
        }
    }
    
    results_path = f'{data_path}/../ppo_experiment_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到 {results_path}")
    
    return results, trainer


if __name__ == '__main__':
    data_path = '/Users/zhangkui/Public/hermes-agent/compute-power-scheduling/data/energy-compute-optimization/dataset'
    results, trainer = run_ppo_experiment(data_path, num_episodes=500)
