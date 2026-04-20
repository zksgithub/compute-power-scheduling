# 算力 - 电力协同优化数学模型

## 基于山东电力 - 数据中心真实数据集

---

## 1. 数据集概览

### 1.1 电力系统（山东）

| 能源类别 | 装机容量 | 占比 |
|----------|----------|------|
| 新能源 | 132,379 MW (132.4 GW) | 35.0% |
| 传统能源 | 216,109 MW (216.1 GW) | 57.2% |
| 核电 | 29,471 MW (29.5 GW) | 7.8% |
| **总计** | **377,959 MW (378.0 GW)** | **100%** |

### 1.2 数据中心

| 资源类型 | 规模 |
|----------|------|
| GPU 节点 | 4,278 个 |
| GPU 总数 | 10,412 卡 |
| GenAI 任务 | 68,195 个 |
| 平均执行时间 | 27.1 秒 |
| 峰值功耗 | 481 MW |

---

## 2. 气象 - 新能源出力模型

### 2.1 风电出力模型

**输入变量**：
- $v(t)$：$t$ 时刻风速（m/s）
- $P_{\text{rated}}$：风机额定功率（MW）
- $v_{\text{cut-in}}$：切入风速（通常 3-4 m/s）
- $v_{\text{rated}}$：额定风速（通常 12-15 m/s）
- $v_{\text{cut-out}}$：切出风速（通常 25 m/s）

**出力模型**：

$$P_{\text{wind}}(t) = \begin{cases}
0, & v(t) < v_{\text{cut-in}} \text{ 或 } v(t) > v_{\text{cut-out}} \\
P_{\text{rated}} \cdot \frac{v(t)^3 - v_{\text{cut-in}}^3}{v_{\text{rated}}^3 - v_{\text{cut-in}}^3}, & v_{\text{cut-in}} \leq v(t) < v_{\text{rated}} \\
P_{\text{rated}}, & v_{\text{rated}} \leq v(t) \leq v_{\text{cut-out}}
\end{cases}$$

**总风电出力**（考虑尾流效应和可用性）：

$$P_{\text{wind,total}}(t) = \eta_{\text{wake}} \cdot \eta_{\text{avail}} \cdot \sum_{i=1}^{N_{\text{turbines}}} P_{\text{wind},i}(t)$$

其中：
- $\eta_{\text{wake}} \approx 0.9$：尾流效应系数
- $\eta_{\text{avail}} \approx 0.95$：风机可用率
- $N_{\text{turbines}}$：风机数量

### 2.2 光伏出力模型

**输入变量**：
- $I(t)$：$t$ 时刻辐照度（W/m²）
- $T(t)$：$t$ 时刻环境温度（°C）
- $P_{\text{STC}}$：标准测试条件下额定功率（MW）
- $I_{\text{STC}} = 1000$ W/m²：标准辐照度
- $T_{\text{STC}} = 25$°C：标准温度
- $\gamma$：温度系数（通常 -0.004~-0.005/°C）

**出力模型**：

$$P_{\text{solar}}(t) = P_{\text{STC}} \cdot \frac{I(t)}{I_{\text{STC}}} \cdot [1 + \gamma \cdot (T(t) - T_{\text{STC}})]$$

**总光伏出力**：

$$P_{\text{solar,total}}(t) = \eta_{\text{inv}} \cdot \eta_{\text{soiling}} \cdot \sum_{j=1}^{N_{\text{panels}}} P_{\text{solar},j}(t)$$

其中：
- $\eta_{\text{inv}} \approx 0.96$：逆变器效率
- $\eta_{\text{soiling}} \approx 0.98$：污损系数

### 2.3 新能源总出力

$$P_{\text{renew}}(t) = P_{\text{wind,total}}(t) + P_{\text{solar,total}}(t)$$

**山东数据**：
- 风电装机：约 70 GW
- 光伏装机：约 50 GW
- 新能源总装机：132.4 GW

---

## 3. 供电模型

### 3.1 电源分类

| 电源类型 | 符号 | 特点 |
|----------|------|------|
| 新能源 | $P_{\text{renew}}(t)$ | 波动性、间歇性、边际成本≈0 |
| 核电 | $P_{\text{nuclear}}(t)$ | 基荷电源、出力稳定 |
| 煤电 | $P_{\text{coal}}(t)$ | 可调节、成本中等 |
| 燃气 | $P_{\text{gas}}(t)$ | 快速响应、成本高 |
| 外购电 | $P_{\text{grid}}(t)$ | 实时电价、碳强度高 |
| 储能放电 | $P_{\text{dis}}(t)$ | 快速响应、效率高 |

### 3.2 供电平衡约束

$$P_{\text{renew}}(t) + P_{\text{nuclear}}(t) + P_{\text{coal}}(t) + P_{\text{gas}}(t) + P_{\text{grid}}(t) + P_{\text{dis}}(t) = P_{\text{load}}(t) + P_{\text{ch}}(t)$$

其中：
- $P_{\text{load}}(t)$：数据中心负载
- $P_{\text{ch}}(t)$：储能充电功率

### 3.3 各类电源约束

**核电（基荷）**：
$$P_{\text{nuclear}}^{\min} \leq P_{\text{nuclear}}(t) \leq P_{\text{nuclear}}^{\max}$$
$$|P_{\text{nuclear}}(t) - P_{\text{nuclear}}(t-1)| \leq R_{\text{nuclear}}$$

**煤电（可调节）**：
$$P_{\text{coal}}^{\min} \leq P_{\text{coal}}(t) \leq P_{\text{coal}}^{\max}$$
$$|P_{\text{coal}}(t) - P_{\text{coal}}(t-1)| \leq R_{\text{coal}}$$

**燃气（快速响应）**：
$$P_{\text{gas}}^{\min} \leq P_{\text{gas}}(t) \leq P_{\text{gas}}^{\max}$$
$$|P_{\text{gas}}(t) - P_{\text{gas}}(t-1)| \leq R_{\text{gas}}$$

**外购电**：
$$0 \leq P_{\text{grid}}(t) \leq P_{\text{grid}}^{\max}$$

**储能系统**：
$$\begin{aligned}
E(t+1) &= E(t) + \eta_{\text{ch}} P_{\text{ch}}(t) \Delta t - \frac{1}{\eta_{\text{dis}}} P_{\text{dis}}(t) \Delta t \\
E^{\min} &\leq E(t) \leq E^{\max} \\
0 &\leq P_{\text{ch}}(t) \leq P_{\text{ch}}^{\max} \\
0 &\leq P_{\text{dis}}(t) \leq P_{\text{dis}}^{\max}
\end{aligned}$$

### 3.4 发电成本模型

**煤电成本**：
$$C_{\text{coal}}(t) = a_{\text{coal}} \cdot P_{\text{coal}}(t)^2 + b_{\text{coal}} \cdot P_{\text{coal}}(t) + c_{\text{coal}}$$

**燃气成本**：
$$C_{\text{gas}}(t) = a_{\text{gas}} \cdot P_{\text{gas}}(t)^2 + b_{\text{gas}} \cdot P_{\text{gas}}(t) + c_{\text{gas}}$$

**外购电成本**（分时电价）：
$$C_{\text{grid}}(t) = \lambda_{\text{elec}}(t) \cdot P_{\text{grid}}(t)$$

其中 $\lambda_{\text{elec}}(t)$ 为 $t$ 时刻电价（元/kWh）

---

## 4. 算力 - 电力消耗模型

### 4.1 GPU 功耗模型

**单 GPU 功耗**：

$$P_{\text{GPU}}(l) = P_{\text{static}} + l \cdot (P_{\text{dynamic}}^{\max} - P_{\text{static}})$$

其中：
- $l \in [0, 1]$：GPU 负载率
- $P_{\text{static}}$：静态功耗（约 50-100W）
- $P_{\text{dynamic}}^{\max}$：满载功耗

**不同 GPU 型号功耗**：

| GPU 型号 | 静态功耗 | 满载功耗 | 数量 |
|----------|----------|----------|------|
| A100-SXM4-80GB | 75W | 400W | 3,456 |
| A10 | 30W | 150W | 2,494 |
| H800 | 75W | 400W | 1,752 |
| GPU-series-1 | 50W | 250W | 1,558 |
| GPU-series-2 | 40W | 200W | 976 |
| A800-SXM4-80GB | 75W | 400W | 176 |

**数据中心 IT 总功耗**：

$$P_{\text{IT}}(t) = \sum_{k=1}^{N_{\text{GPU}}} P_{\text{GPU},k}(l_k(t)) + P_{\text{other}}$$

其中 $P_{\text{other}}$ 为 CPU、内存、网络等其他 IT 设备功耗。

### 4.2 冷却系统功耗模型

**PUE 模型**：

$$\text{PUE}(t) = \frac{P_{\text{total}}(t)}{P_{\text{IT}}(t)} = 1 + \frac{P_{\text{cool}}(t) + P_{\text{lighting}}(t)}{P_{\text{IT}}(t)}$$

**简化线性模型**：

$$\text{PUE}(t) = \alpha + \beta \cdot \frac{P_{\text{IT}}(t)}{P_{\text{IT}}^{\max}}$$

其中：
- $\alpha \approx 1.1$：基础 PUE
- $\beta \approx 0.2$：负载相关系数

**数据中心总功耗**：

$$P_{\text{DC}}(t) = \text{PUE}(t) \cdot P_{\text{IT}}(t)$$

### 4.3 任务功耗模型

**单任务能耗**：

$$E_{\text{task}} = \int_{t_{\text{start}}}^{t_{\text{end}}} P_{\text{GPU}}(l(t)) dt$$

**基于任务类型的平均功耗**（根据数据集统计）：

| 任务类型 | 平均执行时间 | 平均 GPU 负载 | 单任务能耗 |
|----------|--------------|--------------|------------|
| TXT_2_IMG | 27.1 秒 | 85% | 约 0.009 kWh |
| IMG_2_IMG | 25.5 秒 | 80% | 约 0.008 kWh |
| INPAINTING | 30.2 秒 | 90% | 约 0.010 kWh |

---

## 5. 算力 - 电力协同调度模型

### 5.1 决策变量

**连续变量**：
- $P_{\text{coal}}(t), P_{\text{gas}}(t), P_{\text{grid}}(t)$：各电源出力
- $P_{\text{ch}}(t), P_{\text{dis}}(t)$：储能充放电功率
- $l_k(t)$：GPU $k$ 的负载率

**离散变量**：
- $x_j(t) \in \{0, 1\}$：任务 $j$ 在 $t$ 时刻是否执行
- $y_k(t) \in \{0, 1\}$：GPU $k$ 在 $t$ 时刻是否开启

### 5.2 优化目标

**多目标优化**：

$$\min \left[ w_1 \cdot C_{\text{total}} + w_2 \cdot C_{\text{carbon}} - w_3 \cdot \eta_{\text{renew}} \right]$$

**目标 1：总成本最小化**

$$C_{\text{total}} = \sum_{t=1}^{T} \left[ C_{\text{coal}}(t) + C_{\text{gas}}(t) + C_{\text{grid}}(t) \right] \Delta t$$

**目标 2：碳排放最小化**

$$C_{\text{carbon}} = \sum_{t=1}^{T} \left[ \text{CI}_{\text{coal}} \cdot P_{\text{coal}}(t) + \text{CI}_{\text{gas}} \cdot P_{\text{gas}}(t) + \text{CI}_{\text{grid}}(t) \cdot P_{\text{grid}}(t) \right] \Delta t$$

其中 $\text{CI}$ 为碳强度（kgCO₂/kWh）：
- 煤电：约 0.85 kgCO₂/kWh
- 燃气：约 0.45 kgCO₂/kWh
- 电网：约 0.58 kgCO₂/kWh（山东平均）

**目标 3：新能源消纳率最大化**

$$\eta_{\text{renew}} = \frac{\sum_{t=1}^{T} P_{\text{renew,used}}(t)}{\sum_{t=1}^{T} P_{\text{renew,total}}(t)}$$

其中 $P_{\text{renew,used}}(t) = \min(P_{\text{renew}}(t), P_{\text{load}}(t) + P_{\text{ch}}(t))$

### 5.3 约束条件

**1. 供电平衡约束**：
$$P_{\text{renew}}(t) + P_{\text{nuclear}}(t) + P_{\text{coal}}(t) + P_{\text{gas}}(t) + P_{\text{grid}}(t) + P_{\text{dis}}(t) = P_{\text{DC}}(t) + P_{\text{ch}}(t)$$

**2. 电源出力约束**：
$$\begin{aligned}
P_{\text{nuclear}}^{\min} &\leq P_{\text{nuclear}}(t) \leq P_{\text{nuclear}}^{\max} \\
P_{\text{coal}}^{\min} &\leq P_{\text{coal}}(t) \leq P_{\text{coal}}^{\max} \\
P_{\text{gas}}^{\min} &\leq P_{\text{gas}}(t) \leq P_{\text{gas}}^{\max} \\
0 &\leq P_{\text{grid}}(t) \leq P_{\text{grid}}^{\max}
\end{aligned}$$

**3. 爬坡率约束**：
$$|P_{\text{coal}}(t) - P_{\text{coal}}(t-1)| \leq R_{\text{coal}}$$
$$|P_{\text{gas}}(t) - P_{\text{gas}}(t-1)| \leq R_{\text{gas}}$$

**4. 储能约束**：
$$\begin{aligned}
E(t+1) &= E(t) + \eta_{\text{ch}} P_{\text{ch}}(t) \Delta t - \frac{1}{\eta_{\text{dis}}} P_{\text{dis}}(t) \Delta t \\
E^{\min} &\leq E(t) \leq E^{\max} \\
0 &\leq P_{\text{ch}}(t) \leq P_{\text{ch}}^{\max} \\
0 &\leq P_{\text{dis}}(t) \leq P_{\text{dis}}^{\max}
\end{aligned}$$

**5. 任务调度约束**：
$$\sum_{t=\text{arrival}_j}^{\text{deadline}_j} x_j(t) \geq 1, \quad \forall j$$

**6. GPU 资源约束**：
$$\sum_{j} \text{GPU\_req}_j \cdot x_j(t) \leq \sum_{k} \text{GPU\_cap}_k \cdot y_k(t)$$

**7. QoS 约束**：
$$\text{completion}_j \leq \text{deadline}_j + \text{slack}, \quad \forall j$$

---

## 6. 求解方法

### 6.1 混合整数线性规划（MILP）

**适用场景**：小规模问题、离线规划

**优点**：
- 可保证全局最优
- 有成熟的求解器（CPLEX、Gurobi）

**缺点**：
- 大规模问题求解时间长
- 难以处理非线性

### 6.2 模型预测控制（MPC）

**适用场景**：实时调度、滚动优化

**算法流程**：
1. 在 $t$ 时刻，预测未来 $H$ 个时段的负荷、新能源出力、电价
2. 求解优化问题，得到最优控制序列
3. 执行第一个控制动作
4. 在 $t+1$ 时刻，重复步骤 1-3

### 6.3 强化学习（RL）

**适用场景**：不确定性高、在线学习

**状态空间**：
$$s_t = [P_{\text{renew}}(t), \lambda_{\text{elec}}(t), \text{CI}_{\text{grid}}(t), \text{task\_queue}_t, E(t), \text{hour}_t]$$

**动作空间**：
$$a_t = [P_{\text{coal}}(t), P_{\text{gas}}(t), P_{\text{grid}}(t), P_{\text{ch}}(t), P_{\text{dis}}(t), \text{task\_schedule}_t]$$

**奖励函数**：
$$r_t = -\left( w_1 \cdot C_{\text{total}}(t) + w_2 \cdot C_{\text{carbon}}(t) - w_3 \cdot \eta_{\text{renew}}(t) \right)$$

---

## 7. 预期效果

基于山东数据集的仿真结果（预期）：

| 指标 | 基准方法 | 优化方法 | 提升 |
|------|----------|----------|------|
| 新能源消纳率 | 70% | 90%+ | +20% |
| 用电成本 | 100% | 80-85% | -15~-20% |
| 碳排放 | 100% | 75-80% | -20~-25% |
| QoS 违约率 | <0.5% | <1% | 基本持平 |

---

## 8. 数据文件说明

| 文件 | 用途 | 记录数 |
|------|------|--------|
| `shandong_power_classified.csv` | 电源出力建模 | 3,046 |
| `data_trace_processed.csv` | 任务负载建模 | 68,195 |
| `node_info_df.csv` | GPU 功耗建模 | 4,278 |
| `qps.csv` | 负载预测 | 24,627 |
| `pod_gpu_duty_cycle_anon.csv` | GPU 利用率 | 157,417 |

---

*文档版本：v1.0*
*更新时间：2026-04-20*
