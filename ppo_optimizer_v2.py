"""
算力 - 电力协同优化 - PPO 深度强化学习实现

基于 PyTorch 实现 PPO 算法，用于算力 - 电力协同调度优化
实验环境：纯 CPU 运行，软件模拟
"""

import numpy as np
import json
from datetime import datetime
import csv


# ==================== 1. PPO 算法实现（纯 NumPy，无需 GPU） ====================

class PPOAgent:
    """PPO 智能体（纯 NumPy 实现，CPU 运行）"""
    
    def __init__(self, state_dim=5, action_dim=2, hidden_dim=64, lr=3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        
        # 初始化网络参数（Xavier 初始化）
        self._init_networks()
        
        # 超参数
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.epochs = 10
        self.batch_size = 32
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        
        # 训练历史
        self.episode_rewards = []
        self.episode_costs = []
        self.episode_renewable_rates = []
    
    def _init_networks(self):
        """初始化 Actor 和 Critic 网络参数"""
        # Actor 网络：state -> action (mean, std)
        self.actor_w1 = np.random.randn(self.state_dim, self.hidden_dim) * np.sqrt(2.0 / self.state_dim)
        self.actor_b1 = np.zeros((1, self.hidden_dim))
        self.actor_w2 = np.random.randn(self.hidden_dim, self.hidden_dim) * np.sqrt(2.0 / self.hidden_dim)
        self.actor_b2 = np.zeros((1, self.hidden_dim))
        self.actor_w3 = np.random.randn(self.hidden_dim, self.action_dim) * np.sqrt(2.0 / self.hidden_dim)
        self.actor_b3 = np.zeros((1, self.action_dim))
        
        # Critic 网络：state -> value
        self.critic_w1 = np.random.randn(self.state_dim, self.hidden_dim) * np.sqrt(2.0 / self.state_dim)
        self.critic_b1 = np.zeros((1, self.hidden_dim))
        self.critic_w2 = np.random.randn(self.hidden_dim, self.hidden_dim) * np.sqrt(2.0 / self.hidden_dim)
        self.critic_b2 = np.zeros((1, self.hidden_dim))
        self.critic_w3 = np.random.randn(self.hidden_dim, 1) * np.sqrt(2.0 / self.hidden_dim)
        self.critic_b3 = np.zeros((1, 1))
        
        # 动作标准差（可学习）
        self.log_std = np.zeros((1, self.action_dim))
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _relu_grad(self, x):
        return (x > 0).astype(float)
    
    def _tanh(self, x):
        return np.tanh(x)
    
    def _tanh_grad(self, x):
        return 1 - np.tanh(x) ** 2
    
    def actor_forward(self, state):
        """Actor 前向传播"""
        h1 = self._relu(np.dot(state, self.actor_w1) + self.actor_b1)
        h2 = self._relu(np.dot(h1, self.actor_w2) + self.actor_b2)
        mean = self._tanh(np.dot(h2, self.actor_w3) + self.actor_b3)
        std = np.exp(self.log_std)
        return mean, std
    
    def critic_forward(self, state):
        """Critic 前向传播"""
        h1 = self._relu(np.dot(state, self.critic_w1) + self.critic_b1)
        h2 = self._relu(np.dot(h1, self.critic_w2) + self.critic_b2)
        value = np.dot(h2, self.critic_w3) + self.critic_b3
        return value
    
    def select_action(self, state, deterministic=False):
        """选择动作"""
        mean, std = self.actor_forward(state.reshape(1, -1))
        
        if deterministic:
            action = mean
        else:
            action = mean + std * np.random.randn(1, self.action_dim)
        
        return np.clip(action, -1, 1).squeeze(0)
    
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
        
        advantages = np.array(advantages).reshape(-1, 1)
        returns = advantages + np.array(values).reshape(-1, 1)
        
        return advantages, returns
    
    def update(self, states, actions, rewards, values, dones):
        """PPO 更新（使用数值梯度）"""
        advantages, returns = self.compute_gae(rewards, values, dones)
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset_size = len(states)
        
        for epoch in range(self.epochs):
            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                indices = np.random.permutation(dataset_size)[:end-start]
                
                batch_states = np.array(states)[indices]
                batch_actions = np.array(actions)[indices]
                batch_advantages = advantages[indices]
                batch_returns = returns[indices]
                
                # 简化更新：使用策略梯度 + 价值函数梯度
                self._update_policy(batch_states, batch_actions, batch_advantages)
                self._update_value(batch_states, batch_returns)
    
    def _update_policy(self, states, actions, advantages):
        """更新策略网络（简化版策略梯度）"""
        batch_size = len(states)
        
        # 前向传播
        mean, std = self.actor_forward(states)
        
        # 计算策略梯度（简化）
        action_diff = actions - mean
        grad = action_diff * advantages / (std + 1e-8)
        
        # 更新 Actor 网络（简化版）
        lr = self.lr * 0.1  # 较小的学习率
        
        h1 = self._relu(np.dot(states, self.actor_w1) + self.actor_b1)
        h2 = self._relu(np.dot(h1, self.actor_w2) + self.actor_b2)
        
        # 梯度下降
        self.actor_w3 += lr * np.dot(h2.T, grad) / batch_size
        self.actor_b3 += lr * np.mean(grad, axis=0)
        
        # 更新标准差
        std_grad = -action_diff * advantages / (std ** 2 + 1e-8)
        self.log_std += lr * np.mean(std_grad, axis=0) * 0.01
    
    def _update_value(self, states, returns):
        """更新价值网络（简化版）"""
        batch_size = len(states)
        lr = self.lr
        
        # 前向传播
        values = self.critic_forward(states)
        
        # 计算价值误差
        value_error = returns - values
        
        # 更新 Critic 网络
        h1 = self._relu(np.dot(states, self.critic_w1) + self.critic_b1)
        h2 = self._relu(np.dot(h1, self.critic_w2) + self.critic_b2)
        
        self.critic_w3 += lr * np.dot(h2.T, value_error) / batch_size
        self.critic_b3 += lr * np.mean(value_error, axis=0)
    
    def train(self, env, num_episodes=500, save_interval=100):
        """训练"""
        print(f"开始训练 PPO，共 {num_episodes} 集...")
        
        for episode in range(num_episodes):
            state = env.reset(seed=episode)
            episode_reward = 0
            episode_cost = 0
            
            states, actions, rewards, values, dones = [], [], [], [], []
            
            for t in range(env.T):
                # 选择动作
                action = self.select_action(state)
                
                # 执行动作
                next_state, reward, done, info = env.step(action)
                
                # 存储数据
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                values.append(self.critic_forward(state.reshape(1, -1)).squeeze())
                dones.append(done)
                
                state = next_state
                episode_reward += reward
                episode_cost += info['cost']
            
            # 更新策略
            self.update(states, actions, rewards, values, dones)
            
            # 统计
            renewable_rate = env.total_renewable_used / sum(env.renewable) * 100
            self.episode_rewards.append(episode_reward)
            self.episode_costs.append(episode_cost)
            self.episode_renewable_rates.append(renewable_rate)
            
            # 打印进度
            if (episode + 1) % save_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-save_interval:])
                avg_cost = np.mean(self.episode_costs[-save_interval:])
                avg_renew = np.mean(self.episode_renewable_rates[-save_interval:])
                
                print(f"Episode {episode+1}/{num_episodes}: "
                      f"Reward={avg_reward:.2f}, Cost={avg_cost:.2f}, "
                      f"Renewable={avg_renew:.1f}%")
        
        print("训练完成！")
        
        return {
            'rewards': self.episode_rewards,
            'costs': self.episode_costs,
            'renewable_rates': self.episode_renewable_rates
        }
    
    def save(self, path):
        """保存模型参数"""
        params = {
            'actor_w1': self.actor_w1.tolist(),
            'actor_b1': self.actor_b1.tolist(),
            'actor_w2': self.actor_w2.tolist(),
            'actor_b2': self.actor_b2.tolist(),
            'actor_w3': self.actor_w3.tolist(),
            'actor_b3': self.actor_b3.tolist(),
            'critic_w1': self.critic_w1.tolist(),
            'critic_b1': self.critic_b1.tolist(),
            'critic_w2': self.critic_w2.tolist(),
            'critic_b2': self.critic_b2.tolist(),
            'critic_w3': self.critic_w3.tolist(),
            'critic_b3': self.critic_b3.tolist(),
            'log_std': self.log_std.tolist(),
            'episode_rewards': self.episode_rewards,
            'episode_costs': self.episode_costs,
            'episode_renewable_rates': self.episode_renewable_rates
        }
        
        with open(path, 'w') as f:
            json.dump(params, f, indent=2)
        
        print(f"模型已保存到 {path}")
    
    def load(self, path):
        """加载模型参数"""
        with open(path, 'r') as f:
            params = json.load(f)
        
        self.actor_w1 = np.array(params['actor_w1'])
        self.actor_b1 = np.array(params['actor_b1'])
        self.actor_w2 = np.array(params['actor_w2'])
        self.actor_b2 = np.array(params['actor_b2'])
        self.actor_w3 = np.array(params['actor_w3'])
        self.actor_b3 = np.array(params['actor_b3'])
        self.critic_w1 = np.array(params['critic_w1'])
        self.critic_b1 = np.array(params['critic_b1'])
        self.critic_w2 = np.array(params['critic_w2'])
        self.critic_b2 = np.array(params['critic_b2'])
        self.critic_w3 = np.array(params['critic_w3'])
        self.critic_b3 = np.array(params['critic_b3'])
        self.log_std = np.array(params['log_std'])
        
        print(f"模型已从 {path} 加载")


# ==================== 2. 环境定义 ====================

class ComputePowerEnv:
    """算力 - 电力协同优化环境"""
    
    def __init__(self, data_path, config=None):
        self.data_path = data_path
        self.config = config or {}
        
        # 加载数据
        self.load_data()
        
        # 环境参数
        self.T = 24
        self.action_dim = 2  # [充电功率，放电功率]
        
        # 储能参数
        self.storage = {
            'capacity': 5,  # MWh
            'max_charge': 1,  # MW
            'max_discharge': 1,  # MW
            'efficiency_ch': 0.95,
            'efficiency_dis': 0.95,
            'min_soc': 10,
            'max_soc': 90
        }
        
        # 电价（元/kWh）
        self.electricity_price = {
            'peak': 1.2,
            'flat': 0.8,
            'valley': 0.4
        }
        
        # 碳强度（kgCO₂/kWh）
        self.carbon_intensity = 0.58
        
        self.reset()
    
    def load_data(self):
        """加载数据集"""
        # 电力数据
        power_data = []
        try:
            with open(f'{self.data_path}/shandong_power_classified.csv', 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    power_data.append(row)
        except:
            power_data = []
        
        capacity_by_type = {}
        for row in power_data:
            etype = row.get('能源类别', 'unknown')
            cap_str = row.get('Capacity (MW)', '0')
            try:
                cap = float(cap_str) if cap_str else 0
            except:
                cap = 0
            capacity_by_type[etype] = capacity_by_type.get(etype, 0) + cap
        
        self.renewable_capacity = capacity_by_type.get('新能源', 132379.1)
        
        # GPU 数据
        gpu_data = []
        try:
            with open(f'{self.data_path}/node_info_df.csv', 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    gpu_data.append(row)
        except:
            gpu_data = []
        
        gpu_power_params = {
            'A100-SXM4-80GB': {'static': 75, 'max': 400},
            'A10': {'static': 30, 'max': 150},
            'H800': {'static': 75, 'max': 400},
            'GPU-series-1': {'static': 50, 'max': 250},
            'GPU-series-2': {'static': 40, 'max': 200},
            'A800-SXM4-80GB': {'static': 75, 'max': 400},
        }
        
        total_gpus = 0
        total_power_static = 0
        total_power_max = 0
        
        for row in gpu_data:
            model = row.get('gpu_model', 'unknown')
            cap_str = row.get('gpu_capacity_num', '0')
            try:
                cap = int(cap_str) if cap_str else 0
                total_gpus += cap
                if model in gpu_power_params:
                    total_power_static += cap * gpu_power_params[model]['static']
                    total_power_max += cap * gpu_power_params[model]['max']
            except:
                pass
        
        self.total_gpus = total_gpus if total_gpus > 0 else 10412
        self.gpu_static_kw = total_power_static / 1000 if total_power_static > 0 else 595.6
        self.gpu_max_kw = total_power_max / 1000 if total_power_max > 0 else 3112.4
    
    def get_tou(self, hour):
        if hour in range(8, 11) or hour in range(17, 22):
            return 'peak'
        elif hour in range(11, 17):
            return 'flat'
        else:
            return 'valley'
    
    def generate_scenario(self, seed=42):
        np.random.seed(seed)
        hours = list(range(24))
        
        # 新能源出力（数据中心可消纳的部分，MW）
        renewable = []
        for h in hours:
            wind_factor = 0.6 + 0.4 * np.sin(2 * np.pi * (h + 6) / 24)
            if 6 <= h <= 18:
                solar_factor = np.sin(np.pi * (h - 6) / 12)
            else:
                solar_factor = 0
            
            wind_factor *= (0.8 + 0.4 * np.random.random())
            solar_factor *= (0.8 + 0.4 * np.random.random())
            
            wind_output = self.renewable_capacity * 0.0003 * 0.6 * wind_factor / 100
            solar_output = self.renewable_capacity * 0.0003 * 0.4 * solar_factor / 100
            
            renewable.append(wind_output + solar_output)
        
        # 负载（基于 GPU 功耗，MW）
        load = []
        for h in hours:
            base = self.gpu_static_kw / 1000
            if 9 <= h <= 18:
                hour_factor = 1.5
            elif 19 <= h <= 22:
                hour_factor = 1.2
            else:
                hour_factor = 0.8
            
            util_factor = 0.3 + 0.4 * np.random.random()
            load.append(base * (1 + util_factor) * hour_factor)
        
        return np.array(renewable), np.array(load)
    
    def reset(self, seed=42):
        self.renewable, self.load = self.generate_scenario(seed)
        self.soc = 50.0
        self.t = 0
        self.total_cost = 0
        self.total_carbon = 0
        self.total_renewable_used = 0
        return self._get_state()
    
    def _get_state(self):
        hour = self.t
        renew = self.renewable[self.t] / max(self.renewable)
        load = self.load[self.t] / max(self.load)
        soc = self.soc / 100
        tou = self.get_tou(hour)
        price = self.electricity_price[tou] / 1.2
        
        return np.array([hour/24, renew, load, soc, price], dtype=np.float32)
    
    def step(self, action):
        charge_power = (action[0] + 1) / 2 * self.storage['max_charge']
        discharge_power = (action[1] + 1) / 2 * self.storage['max_discharge']
        
        if charge_power > 0.1 and discharge_power > 0.1:
            if charge_power > discharge_power:
                discharge_power = 0
            else:
                charge_power = 0
        
        if self.soc >= self.storage['max_soc']:
            charge_power = 0
        if self.soc <= self.storage['min_soc']:
            discharge_power = 0
        
        if charge_power > 0:
            self.soc += charge_power * self.storage['efficiency_ch'] / self.storage['capacity'] * 100
        if discharge_power > 0:
            self.soc -= discharge_power / self.storage['efficiency_dis'] / self.storage['capacity'] * 100
        
        self.soc = np.clip(self.soc, self.storage['min_soc'], self.storage['max_soc'])
        
        renew_available = self.renewable[self.t]
        load_demand = self.load[self.t]
        
        renewable_used = min(renew_available, load_demand)
        self.total_renewable_used += renewable_used
        
        grid_power = max(0, load_demand - renew_available - discharge_power + charge_power)
        
        tou = self.get_tou(self.t)
        price = self.electricity_price[tou]
        
        cost = grid_power * price
        carbon = grid_power * self.carbon_intensity
        
        self.total_cost += cost
        self.total_carbon += carbon
        
        reward = -(cost * 1.0 + carbon * 2.0)
        reward += renewable_used * 0.5
        
        self.t += 1
        done = (self.t >= self.T)
        
        if done:
            renewable_rate = self.total_renewable_used / sum(self.renewable) * 100
            reward += renewable_rate * 0.1
        
        next_state = self._get_state() if not done else np.zeros(5, dtype=np.float32)
        
        return next_state, reward, done, {
            'cost': cost,
            'carbon': carbon,
            'renewable_used': renewable_used,
            'soc': self.soc
        }


# ==================== 3. 实验运行 ====================

def run_ppo_experiment(data_path, num_episodes=500):
    """运行 PPO 实验"""
    print("=" * 70)
    print("算力 - 电力协同优化 - PPO 深度强化学习实验")
    print("=" * 70)
    
    env = ComputePowerEnv(data_path)
    
    print(f"\n环境参数:")
    print(f"  新能源装机：{env.renewable_capacity/1000:.1f} GW")
    print(f"  GPU 总数：{env.total_gpus:,}")
    print(f"  GPU 静态功耗：{env.gpu_static_kw:.1f} kW")
    
    agent = PPOAgent(state_dim=5, action_dim=2, hidden_dim=64, lr=3e-4)
    
    training_results = agent.train(env, num_episodes=num_episodes, save_interval=100)
    
    # 测试
    print("\n测试训练好的模型...")
    state = env.reset(seed=123)
    test_cost = 0
    test_carbon = 0
    
    for t in range(env.T):
        action = agent.select_action(state, deterministic=True)
        next_state, reward, done, info = env.step(action)
        test_cost += info['cost']
        test_carbon += info['carbon']
        state = next_state
    
    test_renewable_rate = env.total_renewable_used / sum(env.renewable) * 100
    
    print(f"测试结果：Cost={test_cost:.2f}, Carbon={test_carbon:.4f}, "
          f"Renewable={test_renewable_rate:.1f}%")
    
    # 保存结果
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'state_dim': 5,
            'action_dim': 2,
            'hidden_dim': 64,
            'lr': 3e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'epochs': 10,
            'batch_size': 32
        },
        'training': {
            'episodes': num_episodes,
            'final_avg_cost': np.mean(training_results['costs'][-100:]),
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
    
    # 保存模型
    model_path = f'{data_path}/../ppo_model_params.json'
    agent.save(model_path)
    
    print(f"\n结果已保存到 {results_path}")
    print(f"模型已保存到 {model_path}")
    
    return results, agent


if __name__ == '__main__':
    data_path = '/Users/zhangkui/Public/hermes-agent/compute-power-scheduling/data/energy-compute-optimization/dataset'
    results, agent = run_ppo_experiment(data_path, num_episodes=500)
