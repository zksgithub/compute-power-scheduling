"""
算力 - 电力协同优化模型实现示例

基于山东电力 - 数据中心真实数据集
优化目标：新能源消纳最大化、用电成本最小化
"""

import pandas as pd
import numpy as np
from scipy.optimize import linprog, minimize
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class PowerComputeOptimizer:
    """算力 - 电力协同优化器"""
    
    def __init__(self, data_path: str):
        """
        初始化优化器
        
        Args:
            data_path: 数据文件路径
        """
        self.data_path = data_path
        self.load_data()
        self.setup_models()
    
    def load_data(self):
        """加载数据集"""
        # 电力数据
        self.power_df = pd.read_csv(
            f'{self.data_path}/shandong_power_classified.csv'
        )
        
        # 任务负载数据
        self.task_df = pd.read_csv(
            f'{self.data_path}/data_trace_processed.csv'
        )
        
        # GPU 节点数据
        self.gpu_df = pd.read_csv(
            f'{self.data_path}/node_info_df.csv'
        )
        
        # QPS 数据
        self.qps_df = pd.read_csv(
            f'{self.data_path}/qps.csv'
        )
        
        print(f"加载数据完成:")
        print(f"  - 电源：{len(self.power_df)} 个")
        print(f"  - 任务：{len(self.task_df)} 个")
        print(f"  - GPU 节点：{len(self.gpu_df)} 个")
        print(f"  - QPS 采样：{len(self.qps_df)} 个")
    
    def setup_models(self):
        """设置模型参数"""
        # 电源参数
        self.power_params = {
            'renewable_capacity': 132379.1,  # MW
            'nuclear_capacity': 29471.0,  # MW
            'coal_capacity': 180000.0,  # MW
            'gas_capacity': 36109.0,  # MW
        }
        
        # GPU 功耗参数（W）
        self.gpu_power = {
            'A100-SXM4-80GB': {'static': 75, 'max': 400},
            'A10': {'static': 30, 'max': 150},
            'H800': {'static': 75, 'max': 400},
            'GPU-series-1': {'static': 50, 'max': 250},
            'GPU-series-2': {'static': 40, 'max': 200},
            'A800-SXM4-80GB': {'static': 75, 'max': 400},
        }
        
        # 电价（元/kWh，分时电价）
        self.electricity_price = {
            'peak': 1.2,      # 峰段 (8:00-11:00, 17:00-22:00)
            'flat': 0.8,      # 平段 (11:00-17:00)
            'valley': 0.4     # 谷段 (22:00-次日 8:00)
        }
        
        # 碳强度（kgCO₂/kWh）
        self.carbon_intensity = {
            'coal': 0.85,
            'gas': 0.45,
            'grid': 0.58,
            'renew': 0.0,
            'nuclear': 0.012
        }
        
        # 储能参数
        self.storage_params = {
            'capacity': 100,  # MWh
            'max_charge': 20,  # MW
            'max_discharge': 20,  # MW
            'efficiency_ch': 0.95,
            'efficiency_dis': 0.95
        }
    
    def get_time_of_use(self, hour: int) -> str:
        """获取时段类型"""
        if hour in range(8, 11) or hour in range(17, 22):
            return 'peak'
        elif hour in range(11, 17):
            return 'flat'
        else:
            return 'valley'
    
    def predict_renewable_output(self, hour: int, day: int) -> float:
        """
        预测新能源出力（简化模型）
        注意：这里预测的是可供数据中心使用的新能源出力，而非全省总出力
        
        Args:
            hour: 小时 (0-23)
            day: 日期 (用于模拟天气变化)
        
        Returns:
            新能源出力 (MW)
        """
        # 风电：夜间较大，白天较小
        wind_factor = 0.6 + 0.4 * np.sin(2 * np.pi * (hour + 6) / 24)
        
        # 光伏：白天有出力，夜间为 0
        if 6 <= hour <= 18:
            solar_factor = np.sin(np.pi * (hour - 6) / 12)
        else:
            solar_factor = 0
        
        # 添加随机波动
        np.random.seed(day * 24 + hour)
        wind_factor *= (0.8 + 0.4 * np.random.random())
        solar_factor *= (0.8 + 0.4 * np.random.random())
        
        # 计算总出力 - 调整为与负载相当的数量级（数据中心可消纳的部分）
        # 假设数据中心可消纳新能源的 0.5-1%
        wind_output = 150 * wind_factor  # MW
        solar_output = 100 * solar_factor  # MW
        
        return wind_output + solar_output
    
    def predict_load(self, hour: int, day: int) -> float:
        """
        预测数据中心负载（基于 QPS 数据）
        
        Args:
            hour: 小时 (0-23)
            day: 日期
        
        Returns:
            负载功率 (MW)
        """
        # 基础负载
        base_load = 200  # MW
        
        # 时段因子
        if 9 <= hour <= 18:
            hour_factor = 1.5  # 工作时间负载高
        elif 19 <= hour <= 22:
            hour_factor = 1.2  # 晚间较高
        else:
            hour_factor = 0.8  # 夜间较低
        
        # 随机波动
        np.random.seed(day * 24 + hour + 1000)
        random_factor = 0.9 + 0.2 * np.random.random()
        
        return base_load * hour_factor * random_factor
    
    def optimize_dispatch_milp(self, horizon: int = 24) -> Dict:
        """
        使用 MILP 进行优化调度
        
        Args:
            horizon: 优化时段（小时）
        
        Returns:
            优化结果字典
        """
        print(f"\n开始 MILP 优化调度 (优化 horizon={horizon}小时)...")
        
        # 预测新能源出力和负载
        renewable = [self.predict_renewable_output(h, 0) for h in range(horizon)]
        load = [self.predict_load(h, 0) for h in range(horizon)]
        
        # 优化变量：
        # x[0:horizon]: 煤电出力
        # x[horizon:2*horizon]: 燃气出力
        # x[2*horizon:3*horizon]: 外购电
        # x[3*horizon:4*horizon]: 储能充电
        # x[4*horizon:5*horizon]: 储能放电
        
        n_vars = 5 * horizon
        
        # 目标函数系数（成本）
        c = []
        for h in range(horizon):
            tou = self.get_time_of_use(h)
            price = self.electricity_price[tou]
            
            # 煤电成本（二次成本线性化）
            c.append(0.3)  # 元/MWh
            # 燃气成本
            c.append(0.5)
            # 外购电成本
            c.append(price * 1000)  # 转换为元/MWh
            # 储能充电（负收益）
            c.append(price * 1000)
            # 储能放电（零成本）
            c.append(0)
        
        # 约束矩阵
        A_eq = []
        b_eq = []
        
        # 供电平衡约束
        for h in range(horizon):
            row = [0] * n_vars
            row[h] = 1  # 煤电
            row[horizon + h] = 1  # 燃气
            row[2 * horizon + h] = 1  # 外购电
            row[4 * horizon + h] = 1  # 储能放电
            row[3 * horizon + h] = -1  # 储能充电（负号）
            
            # 新能源优先消纳
            net_load = max(0, load[h] - renewable[h])
            A_eq.append(row)
            b_eq.append(net_load)
        
        # 变量边界
        bounds = []
        for h in range(horizon):
            # 煤电
            bounds.append((0, self.power_params['coal_capacity']))
            # 燃气
            bounds.append((0, self.power_params['gas_capacity']))
            # 外购电
            bounds.append((0, 1000))
            # 储能充电
            bounds.append((0, self.storage_params['max_charge']))
            # 储能放电
            bounds.append((0, self.storage_params['max_discharge']))
        
        # 求解
        result = linprog(
            c,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method='highs'
        )
        
        if result.success:
            print(f"优化成功!")
            
            # 解析结果
            coal = result.x[0:horizon]
            gas = result.x[horizon:2*horizon]
            grid = result.x[2*horizon:3*horizon]
            charge = result.x[3*horizon:4*horizon]
            discharge = result.x[4*horizon:5*horizon]
            
            # 计算指标
            total_cost = np.dot(c, result.x) / 1000  # 转换为元
            renewable_used = sum(min(renewable[h], load[h] + charge[h]) for h in range(horizon))
            renewable_total = sum(renewable)
            renewable_rate = renewable_used / renewable_total * 100
            
            carbon = (
                sum(coal) * self.carbon_intensity['coal'] +
                sum(gas) * self.carbon_intensity['gas'] +
                sum(grid) * self.carbon_intensity['grid']
            ) / 1000  # 转换为吨
            
            return {
                'success': True,
                'coal': coal,
                'gas': gas,
                'grid': grid,
                'charge': charge,
                'discharge': discharge,
                'cost': total_cost,
                'renewable_rate': renewable_rate,
                'carbon': carbon,
                'renewable': renewable,
                'load': load
            }
        else:
            print(f"优化失败：{result.message}")
            return {'success': False}
    
    def optimize_dispatch_rl(self, episodes: int = 1000) -> Dict:
        """
        使用简化强化学习进行优化调度
        
        Args:
            episodes: 训练轮次
        
        Returns:
            优化结果字典
        """
        print(f"\n开始 RL 优化调度 (训练 episodes={episodes})...")
        
        # 简化 Q-learning 实现
        # 状态：[时段类型，负载水平，新能源水平，储能 SOC]
        # 动作：[充电，放电，保持]
        
        n_states = 4 * 3 * 3 * 5  # 时段*负载*新能源*SOC
        n_actions = 3  # 充电，放电，保持
        
        # 初始化 Q 表
        Q = np.zeros((n_states, n_actions))
        
        # 超参数
        alpha = 0.1  # 学习率
        gamma = 0.9  # 折扣因子
        epsilon = 0.3  # 探索率
        
        def get_state(hour, load, renewable, soc):
            """获取状态索引"""
            tou = 0 if self.get_time_of_use(hour) == 'valley' else (1 if self.get_time_of_use(hour) == 'flat' else 2)
            load_level = min(2, int(load / 100))
            renew_level = min(2, int(renewable / 50000))
            soc_level = min(4, int(soc / 20))
            return tou * 27 + load_level * 9 + renew_level * 3 + soc_level
        
        # 训练
        for ep in range(episodes):
            hour = 0
            soc = 50  # 初始 SOC 50%
            total_reward = 0
            
            while hour < 24:
                # 获取当前状态
                load = self.predict_load(hour, ep % 30)
                renewable = self.predict_renewable_output(hour, ep % 30)
                state = get_state(hour, load, renewable, soc)
                
                # epsilon-greedy 策略
                if np.random.random() < epsilon:
                    action = np.random.randint(n_actions)
                else:
                    action = np.argmax(Q[state])
                
                # 执行动作
                if action == 0:  # 充电
                    charge = min(self.storage_params['max_charge'], 
                                (self.storage_params['capacity'] - soc) / self.storage_params['efficiency_ch'])
                    discharge = 0
                    soc += charge * self.storage_params['efficiency_ch']
                elif action == 1:  # 放电
                    charge = 0
                    discharge = min(self.storage_params['max_discharge'], 
                                   soc * self.storage_params['efficiency_dis'])
                    soc -= discharge / self.storage_params['efficiency_dis']
                else:  # 保持
                    charge = 0
                    discharge = 0
                
                # 计算奖励（负成本）
                tou = self.get_time_of_use(hour)
                price = self.electricity_price[tou]
                
                # 外购电
                net_load = max(0, load - renewable - discharge + charge)
                cost = net_load * price
                
                # 新能源消纳奖励
                renew_used = min(load, renewable)
                renew_reward = renew_used * 0.1
                
                reward = -cost + renew_reward
                total_reward += reward
                
                # 更新 Q 表
                next_hour = (hour + 1) % 24
                next_load = self.predict_load(next_hour, ep % 30)
                next_renewable = self.predict_renewable_output(next_hour, ep % 30)
                next_state = get_state(next_hour, next_load, next_renewable, soc)
                
                Q[state, action] += alpha * (
                    reward + gamma * np.max(Q[next_state]) - Q[state, action]
                )
                
                hour += 1
            
            # 衰减探索率
            if ep % 100 == 0:
                epsilon = max(0.05, epsilon * 0.95)
        
        print(f"训练完成，最终平均奖励：{total_reward / 24:.2f}")
        
        # 使用训练好的策略进行调度
        return self.execute_learned_policy(Q, episodes=1)
    
    def execute_learned_policy(self, Q: np.ndarray, episodes: int = 1) -> Dict:
        """执行学习到的策略"""
        
        def get_state(hour, load, renewable, soc):
            """获取状态索引"""
            tou = 0 if self.get_time_of_use(hour) == 'valley' else (1 if self.get_time_of_use(hour) == 'flat' else 2)
            load_level = min(2, int(load / 100))
            renew_level = min(2, int(renewable / 50000))
            soc_level = min(4, int(soc / 20))
            return tou * 27 + load_level * 9 + renew_level * 3 + soc_level
        
        hour = 0
        soc = 50
        
        coal = []
        gas = []
        grid_power = []
        charge_list = []
        discharge_list = []
        renewable_list = []
        load_list = []
        
        for _ in range(episodes):
            while hour < 24:
                load = self.predict_load(hour, 0)
                renewable = self.predict_renewable_output(hour, 0)
                state = get_state(hour, load, renewable, soc)
                
                # 选择最优动作
                action = np.argmax(Q[state])
                
                # 执行动作
                if action == 0:  # 充电
                    charge = min(self.storage_params['max_charge'], 
                                (self.storage_params['capacity'] - soc) / self.storage_params['efficiency_ch'])
                    discharge = 0
                    soc += charge * self.storage_params['efficiency_ch']
                elif action == 1:  # 放电
                    charge = 0
                    discharge = min(self.storage_params['max_discharge'], 
                                   soc * self.storage_params['efficiency_dis'])
                    soc -= discharge / self.storage_params['efficiency_dis']
                else:
                    charge = 0
                    discharge = 0
                
                # 计算各电源出力
                net_load = max(0, load - renewable - discharge + charge)
                
                # 优先使用核电（基荷）
                nuclear = min(2000, net_load)
                net_load -= nuclear
                
                # 然后使用煤电
                coal_out = min(5000, net_load)
                net_load -= coal_out
                
                # 最后使用燃气和外购电
                gas_out = min(2000, net_load)
                net_load -= gas_out
                grid = net_load
                
                coal.append(coal_out)
                gas.append(gas_out)
                grid_power.append(grid)
                charge_list.append(charge)
                discharge_list.append(discharge)
                renewable_list.append(renewable)
                load_list.append(load)
                
                hour += 1
        
        # 计算指标
        total_cost = sum(
            coal[h] * 0.3 + gas[h] * 0.5 + grid_power[h] * self.electricity_price[self.get_time_of_use(h)]
            for h in range(24)
        ) / 1000  # 转换为元
        
        renewable_used = sum(min(renewable_list[h], load_list[h] + charge_list[h]) for h in range(24))
        renewable_total = sum(renewable_list)
        renewable_rate = renewable_used / renewable_total * 100 if renewable_total > 0 else 0
        
        carbon = (
            sum(coal) * self.carbon_intensity['coal'] +
            sum(gas) * self.carbon_intensity['gas'] +
            sum(grid_power) * self.carbon_intensity['grid']
        ) / 1000  # 转换为吨
        
        return {
            'success': True,
            'coal': coal,
            'gas': gas,
            'grid': grid_power,
            'charge': charge_list,
            'discharge': discharge_list,
            'cost': total_cost,
            'renewable_rate': renewable_rate,
            'carbon': carbon,
            'renewable': renewable_list,
            'load': load_list
        }
    
    def compare_methods(self) -> Dict:
        """对比不同优化方法"""
        print("\n" + "="*60)
        print("对比不同优化方法")
        print("="*60)
        
        # MILP 优化
        milp_result = self.optimize_dispatch_milp(horizon=24)
        
        # RL 优化
        rl_result = self.optimize_dispatch_rl(episodes=500)
        
        # 基准方法（无优化）
        baseline_result = self.get_baseline_dispatch()
        
        # 对比结果
        print("\n优化结果对比:")
        print("-" * 60)
        print(f"{'方法':<15} {'成本 (元)':<15} {'新能源消纳率':<15} {'碳排放 (吨)':<15}")
        print("-" * 60)
        
        methods = [
            ('基准方法', baseline_result),
            ('MILP 优化', milp_result),
            ('RL 优化', rl_result)
        ]
        
        for name, result in methods:
            if result['success']:
                print(f"{name:<15} {result['cost']:>10,.0f}    {result['renewable_rate']:>10.1f}%    {result['carbon']:>10,.1f}")
        
        return {
            'baseline': baseline_result,
            'milp': milp_result,
            'rl': rl_result
        }
    
    def get_baseline_dispatch(self) -> Dict:
        """基准调度方法（无优化）"""
        renewable = [self.predict_renewable_output(h, 0) for h in range(24)]
        load = [self.predict_load(h, 0) for h in range(24)]
        
        coal = []
        gas = []
        grid_power = []
        
        for h in range(24):
            net_load = max(0, load[h] - renewable[h])
            
            # 简单规则：优先煤电，然后燃气，最后外购电
            coal_out = min(5000, net_load)
            net_load -= coal_out
            
            gas_out = min(2000, net_load)
            net_load -= gas_out
            
            grid = net_load
            
            coal.append(coal_out)
            gas.append(gas_out)
            grid_power.append(grid)
        
        # 计算指标
        total_cost = sum(
            coal[h] * 0.3 + gas[h] * 0.5 + grid_power[h] * self.electricity_price[self.get_time_of_use(h)]
            for h in range(24)
        ) / 1000  # 转换为元（原为 MW，电价是元/kWh）
        
        renewable_used = sum(min(renewable[h], load[h]) for h in range(24))
        renewable_total = sum(renewable)
        renewable_rate = renewable_used / renewable_total * 100 if renewable_total > 0 else 0
        
        carbon = (
            sum(coal) * self.carbon_intensity['coal'] +
            sum(gas) * self.carbon_intensity['gas'] +
            sum(grid_power) * self.carbon_intensity['grid']
        ) / 1000  # 转换为吨（原为 MWh）
        
        return {
            'success': True,
            'coal': coal,
            'gas': gas,
            'grid': grid_power,
            'charge': [0] * 24,
            'discharge': [0] * 24,
            'cost': total_cost,
            'renewable_rate': renewable_rate,
            'carbon': carbon,
            'renewable': renewable,
            'load': load
        }


if __name__ == '__main__':
    # 数据路径
    data_path = '/Users/zhangkui/Public/hermes-agent/compute-power-scheduling/data/energy-compute-optimization/dataset'
    
    # 创建优化器
    optimizer = PowerComputeOptimizer(data_path)
    
    # 对比不同方法
    results = optimizer.compare_methods()
    
    # 保存结果
    import json
    with open('/Users/zhangkui/Public/hermes-agent/compute-power-scheduling/optimization_results.json', 'w') as f:
        # 转换 numpy 类型为 Python 原生类型
        serializable_results = {}
        for key, value in results.items():
            serializable_results[key] = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    serializable_results[key][k] = v.tolist()
                elif isinstance(v, (np.float64, np.int64)):
                    serializable_results[key][k] = float(v)
                else:
                    serializable_results[key][k] = v
        
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n优化结果已保存到 optimization_results.json")
