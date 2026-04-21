"""
算力 - 电力协同优化 - 真实实验（基于真实数据集参数）

修正：
1. 使用真实 GPU 功耗参数计算负载
2. 使用真实新能源装机容量
3. 修正数量级匹配问题
"""

import csv
import numpy as np
import json
from datetime import datetime


# ==================== 1. 真实数据加载 ====================

def load_real_data(data_path):
    """从真实数据集加载参数"""
    
    # 电力数据
    power_data = []
    with open(f'{data_path}/shandong_power_classified.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            power_data.append(row)
    
    capacity_by_type = {}
    for row in power_data:
        etype = row.get('能源类别', 'unknown')
        cap_str = row.get('Capacity (MW)', '0')
        try:
            cap = float(cap_str) if cap_str else 0
        except:
            cap = 0
        capacity_by_type[etype] = capacity_by_type.get(etype, 0) + cap
    
    # GPU 数据
    gpu_data = []
    with open(f'{data_path}/node_info_df.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gpu_data.append(row)
    
    # GPU 功耗参数（W）
    gpu_power_params = {
        'A100-SXM4-80GB': {'static': 75, 'max': 400},
        'A10': {'static': 30, 'max': 150},
        'H800': {'static': 75, 'max': 400},
        'GPU-series-1': {'static': 50, 'max': 250},
        'GPU-series-2': {'static': 40, 'max': 200},
        'A800-SXM4-80GB': {'static': 75, 'max': 400},
    }
    
    # 计算 GPU 总功耗
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
    
    return {
        'renewable_capacity_mw': capacity_by_type.get('新能源', 132379.1),
        'nuclear_capacity_mw': capacity_by_type.get('核电', 29471.0),
        'conventional_capacity_mw': capacity_by_type.get('传统能源', 216109.0),
        'total_gpus': total_gpus,
        'gpu_static_kw': total_power_static / 1000,
        'gpu_max_kw': total_power_max / 1000,
        'gpu_power_params': gpu_power_params
    }


# ==================== 2. 环境定义（修正参数） ====================

class RealComputePowerEnv:
    """基于真实参数的算力 - 电力协同优化环境"""
    
    def __init__(self, data):
        self.data = data
        
        # 真实参数
        self.renewable_capacity = data['renewable_capacity_mw']  # 132.4 GW
        self.total_gpus = data['total_gpus']  # 10,412
        self.gpu_static_kw = data['gpu_static_kw']  # 595.6 kW
        self.gpu_max_kw = data['gpu_max_kw']  # 3112.4 kW
        
        # 数据中心负载参数（基于真实 GPU 功耗）
        # 假设平均利用率 40-60%
        self.base_load_mw = self.gpu_static_kw / 1000 * 1.5  # 约 0.9 MW
        self.peak_load_mw = self.gpu_max_kw / 1000 * 0.6  # 约 1.9 MW
        
        # 新能源消纳参数（数据中心可消纳的比例）
        # 调整使新能源与负载匹配
        self.renewable_share = 0.0003  # 0.03%（使新能源与负载相当）
        
        # 储能参数
        self.storage = {
            'capacity': 5,  # MWh（与负载匹配）
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
        
        self.T = 24
        self.reset()
    
    def get_tou(self, hour):
        if hour in range(8, 11) or hour in range(17, 22):
            return 'peak'
        elif hour in range(11, 17):
            return 'flat'
        else:
            return 'valley'
    
    def generate_scenario(self, seed=42):
        """生成 24 小时场景（基于真实参数）"""
        np.random.seed(seed)
        hours = list(range(24))
        
        # 新能源出力（数据中心可消纳的部分，MW）
        renewable = []
        for h in hours:
            # 风电：夜间较大
            wind_factor = 0.6 + 0.4 * np.sin(2 * np.pi * (h + 6) / 24)
            # 光伏：白天有出力
            if 6 <= h <= 18:
                solar_factor = np.sin(np.pi * (h - 6) / 12)
            else:
                solar_factor = 0
            
            # 添加波动
            wind_factor *= (0.8 + 0.4 * np.random.random())
            solar_factor *= (0.8 + 0.4 * np.random.random())
            
            # 数据中心可消纳的新能源（MW）
            wind_output = self.renewable_capacity * self.renewable_share * 0.6 * wind_factor / 100
            solar_output = self.renewable_capacity * self.renewable_share * 0.4 * solar_factor / 100
            
            renewable.append(wind_output + solar_output)
        
        # 负载（基于真实 GPU 功耗，MW）
        load = []
        for h in hours:
            # 基础负载（静态功耗）
            base = self.gpu_static_kw / 1000  # MW
            
            # 时段因子（基于 QPS 模式）
            if 9 <= h <= 18:
                hour_factor = 1.5  # 工作时间
            elif 19 <= h <= 22:
                hour_factor = 1.2  # 晚间
            else:
                hour_factor = 0.8  # 夜间
            
            # 动态负载（GPU 利用率）
            util_factor = 0.3 + 0.4 * np.random.random()  # 30-70%
            
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
        """执行动作"""
        # 动作：[充电功率，放电功率]
        charge_power = (action[0] + 1) / 2 * self.storage['max_charge']
        discharge_power = (action[1] + 1) / 2 * self.storage['max_discharge']
        
        # 互斥约束
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
        
        # 奖励：负的成本 + 碳排放 + 新能源消纳奖励
        reward = -(cost * 1.0 + carbon * 2.0)
        reward += renewable_used * 0.5  # 新能源消纳奖励
        
        # 更新时段
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


# ==================== 3. 优化方法实现 ====================

def run_rule_based(env, renewable, load):
    """Rule-based：新能源优先"""
    T = len(renewable)
    cost = 0
    carbon = 0
    renewable_used = 0
    
    for t in range(T):
        tou = env.get_tou(t)
        price = env.electricity_price[tou]
        
        used = min(renewable[t], load[t])
        renewable_used += used
        
        grid_power = max(0, load[t] - renewable[t])
        
        cost += grid_power * price
        carbon += grid_power * env.carbon_intensity
    
    renewable_rate = renewable_used / sum(renewable) * 100 if sum(renewable) > 0 else 0
    
    return {'cost': cost, 'renewable_rate': renewable_rate, 'carbon': carbon / 1000}


def run_greedy(env, renewable, load):
    """Greedy：谷段充电、峰段放电"""
    T = len(renewable)
    cost = 0
    carbon = 0
    renewable_used = 0
    soc = 50
    
    for t in range(T):
        tou = env.get_tou(t)
        price = env.electricity_price[tou]
        
        # 储能决策
        if soc < 80 and tou == 'valley':
            charge = min(env.storage['max_charge'], (env.storage['capacity'] - soc) / env.storage['efficiency_ch'])
            discharge = 0
            soc += charge * env.storage['efficiency_ch']
        elif soc > 20 and tou == 'peak':
            discharge = min(env.storage['max_discharge'], soc * env.storage['efficiency_dis'])
            charge = 0
            soc -= discharge / env.storage['efficiency_dis']
        else:
            charge = 0
            discharge = 0
        
        # 新能源消纳（只计算用于负载的部分）
        used = min(renewable[t], load[t])
        renewable_used += used
        
        # 电网购电
        grid_power = max(0, load[t] - renewable[t] - discharge + charge)
        
        cost += grid_power * price
        carbon += grid_power * env.carbon_intensity
    
    # 消纳率：实际使用的新能源 / 总可用新能源
    renewable_rate = renewable_used / sum(renewable) * 100 if sum(renewable) > 0 else 0
    
    return {'cost': cost, 'renewable_rate': renewable_rate, 'carbon': carbon / 1000}


def run_mpc(env, renewable, load, horizon=12):
    """MPC：滚动优化"""
    T = len(renewable)
    cost = 0
    carbon = 0
    renewable_used = 0
    soc = 50
    
    for t in range(T):
        tou = env.get_tou(t)
        price = env.electricity_price[tou]
        
        # 预测
        end = min(t + horizon, T)
        avg_renew = np.mean(renewable[t:end])
        avg_load = np.mean(load[t:end])
        
        # 储能决策
        if soc < 80 and (price == env.electricity_price['valley'] or avg_renew > avg_load * 1.2):
            charge = min(env.storage['max_charge'], (env.storage['capacity'] - soc) / env.storage['efficiency_ch'])
            discharge = 0
            soc += charge * env.storage['efficiency_ch']
        elif soc > 20 and (price == env.electricity_price['peak'] or avg_renew < avg_load * 0.8):
            discharge = min(env.storage['max_discharge'], soc * env.storage['efficiency_dis'])
            charge = 0
            soc -= discharge / env.storage['efficiency_dis']
        else:
            charge = 0
            discharge = 0
        
        # 新能源消纳（只计算用于负载的部分）
        used = min(renewable[t], load[t])
        renewable_used += used
        
        # 电网购电
        grid_power = max(0, load[t] - renewable[t] - discharge + charge)
        
        cost += grid_power * price
        carbon += grid_power * env.carbon_intensity
    
    # 消纳率：实际使用的新能源 / 总可用新能源
    renewable_rate = renewable_used / sum(renewable) * 100 if sum(renewable) > 0 else 0
    
    return {'cost': cost, 'renewable_rate': renewable_rate, 'carbon': carbon / 1000}


# ==================== 4. 主实验 ====================

if __name__ == '__main__':
    data_path = '/Users/zhangkui/Public/hermes-agent/compute-power-scheduling/data/energy-compute-optimization/dataset'
    
    print("=" * 70)
    print("算力 - 电力协同优化 - 真实实验（基于真实数据集参数）")
    print("=" * 70)
    
    # 加载真实数据
    data = load_real_data(data_path)
    
    print(f"\n【真实数据集参数】")
    print(f"  新能源装机：{data['renewable_capacity_mw']/1000:.1f} GW")
    print(f"  GPU 总数：{data['total_gpus']:,}")
    print(f"  GPU 静态功耗：{data['gpu_static_kw']:.1f} kW")
    print(f"  GPU 满载功耗：{data['gpu_max_kw']:.1f} kW")
    
    # 创建环境
    env = RealComputePowerEnv(data)
    
    # 生成场景
    renewable, load = env.generate_scenario(seed=42)
    
    print(f"\n【24 小时场景】")
    print(f"  新能源出力：{min(renewable):.3f} - {max(renewable):.3f} MW")
    print(f"  负载：{min(load):.3f} - {max(load):.3f} MW")
    
    # 运行实验
    print("\n【运行优化实验】")
    
    results = {}
    
    print("  Rule-based...")
    results['Rule-based'] = run_rule_based(env, renewable, load)
    
    print("  Greedy...")
    results['Greedy'] = run_greedy(env, renewable, load)
    
    print("  MPC...")
    results['MPC'] = run_mpc(env, renewable, load)
    
    # 打印结果
    print("\n" + "=" * 70)
    print("实验结果对比")
    print("=" * 70)
    print(f"{'方法':<15} {'成本 (元/天)':<15} {'消纳率 (%)':<15} {'碳排放 (吨/天)':<15}")
    print("-" * 70)
    
    for method, res in results.items():
        print(f"{method:<15} {res['cost']:>10.3f}     {res['renewable_rate']:>10.1f}    {res['carbon']:>10.4f}")
    
    print("-" * 70)
    
    # 计算提升
    baseline = results['Rule-based']
    print(f"\n相比 Rule-based 的提升:")
    for method, res in results.items():
        if method != 'Rule-based':
            cost_improve = (baseline['cost'] - res['cost']) / baseline['cost'] * 100
            renew_improve = res['renewable_rate'] - baseline['renewable_rate']
            carbon_improve = (baseline['carbon'] - res['carbon']) / baseline['carbon'] * 100
            print(f"  {method}: 成本 {cost_improve:+.1f}%, 消纳率 {renew_improve:+.1f}%, 碳排放 {carbon_improve:+.1f}%")
    
    # 保存结果
    output = {
        'timestamp': datetime.now().isoformat(),
        'dataset': data,
        'scenario': {
            'renewable_range_mw': [min(renewable), max(renewable)],
            'load_range_mw': [min(load), max(load)]
        },
        'results': results
    }
    
    with open(f'{data_path}/../real_experiment_v2.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到 real_experiment_v2.json")
