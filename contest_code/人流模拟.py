import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import seaborn as sns
from tqdm import tqdm
from MathModels.Plot.styles import mp_seaborn_light
import pandas as pd
from scipy import stats
from SALib.sample import saltelli
from SALib.analyze import sobol
from tqdm import tqdm

# ====================== 增强参数设置 ======================
# 材料属性
k = 0.01  # 磨损比例常数
H0 = 60.0  # 初始材料硬度 (Pa)
alpha = 0.1  # 人流量对接触压力的线性系数
beta = 0.1  # 基础接触压力系数
v0 = 1.2  # 基础行走速度 (m/s)
gamma = 0.05  # 拥挤衰减系数 (1/人)

# 环境基准参数
base_temperature = 25.0  # 基准温度 (°C)
base_humidity = 0.6  # 基准湿度 (0-1)
temp_noise_std = 2.0  # 温度波动标准差
humidity_noise_std = 0.1  # 湿度波动标准差

# 模拟参数
T = 24.0  # 总模拟时间 (小时)
dt = 0.1  # 时间离散化步长
num_simulations = 500  # 蒙特卡洛模拟次数


# 动态人流强度函数（含昼夜模式）
def lambda_func(s):
    """时变人流强度函数"""
    hour = s % 24
    if 8 <= hour < 10:  # 早高峰
        return 8.0 + 2 * np.sin(2 * np.pi * hour / 24)
    elif 17 <= hour < 19:  # 晚高峰
        return 6.0 + 1.5 * np.sin(2 * np.pi * hour / 24)
    else:  # 平峰
        return 3.0 + 0.5 * np.sin(2 * np.pi * hour / 24)


# ====================== 预计算模块 ======================
# 预计算累积强度函数Lambda(τ)
tau_values = np.linspace(0, T, 1000)
Lambda_values = np.zeros_like(tau_values)

print("预计算累积强度函数...")
for i, tau in enumerate(tqdm(tau_values)):
    Lambda_values[i], _ = quad(lambda_func, 0, tau, limit=1000)


def get_Lambda(tau):
    """带线性插值的累积强度查询"""
    return np.interp(tau, tau_values, Lambda_values)


# ====================== 增强材料模型 ======================
def H(t):
    """时变材料硬度模型（含老化效应）"""
    return H0 * np.exp(-0.001 * t)  # 指数衰减


# 维修事件配置（时间单位：小时）
repair_events = [(6.0, 0.2), (18.0, 0.3)]  # 早晨维护  # 傍晚维护


# ====================== 动态环境生成器 ======================
def get_environment(t):
    """生成动态环境参数（温度/湿度）"""
    # 昼夜温度波动（基准±5°C）
    daily_temp = base_temperature + 5 * np.sin(2 * np.pi * t / 24)
    temp = daily_temp + np.random.normal(0, temp_noise_std)

    # 湿度波动（基准±0.2）
    humidity = base_humidity + np.random.normal(0, humidity_noise_std)

    # 约束合理范围
    temp = np.clip(temp, 15.0, 35.0)
    humidity = np.clip(humidity, 0.3, 0.9)

    return temp, humidity


def env_factor(temp, humidity):
    """非线性环境因子模型"""
    # 温度响应：Sigmoid软化效应（28°C为拐点）
    temp_effect = 1 + 1.5 / (1 + np.exp(-0.5 * (temp - 28)))

    # 湿度响应：二次腐蚀效应（0.6为临界点）
    humidity_effect = 1 + 0.2 * (humidity - 0.6) + 0.4 * (humidity - 0.6) ** 2

    return temp_effect * humidity_effect


# ====================== 理论磨损模型 ======================
def integrand(tau):
    """增强型被积函数（含稳定性控制）"""
    current_H = H(tau)
    Lambda = get_Lambda(tau)

    # 环境参数动态计算
    temp, humidity = get_environment(tau)
    env = env_factor(temp, humidity)

    # 指数项稳定性控制
    exponent = Lambda * (np.exp(-gamma) - 1)
    if exponent < -100:
        exp_term = 0.0
    else:
        exp_term = np.exp(exponent)

    term1 = alpha * Lambda * np.exp(-gamma) * exp_term
    term2 = beta * exp_term
    return (k * v0 / current_H) * env * (term1 + term2)


def calculate_theoretical_wear():
    """分段积分理论计算"""
    E_Q = 0.0
    split_points = sorted([e[0] for e in repair_events] + [T])
    prev_time = 0.0

    for t_end in split_points:
        if t_end <= prev_time:
            continue

        # 自适应区间分割（最大2小时）
        num_sub = max(1, int((t_end - prev_time) / 2.0))
        sub_times = np.linspace(prev_time, t_end, num_sub + 1)

        # 分段积分
        for i in range(num_sub):
            a, b = sub_times[i], sub_times[i + 1]
            partial, _ = quad(integrand, a, b, limit=1000, epsabs=1e-6)
            E_Q += partial

        # 应用维修效果
        for t_repair, eff in repair_events:
            if abs(t_end - t_repair) < 1e-6:
                E_Q *= 1 - eff

        prev_time = t_end

    return E_Q


def run_single_simulation(**params):
    """带参数注入的单个模拟流程"""
    global k, H0, alpha, beta, v0, gamma  # 声明为全局变量以便修改

    # 保存原始参数值
    original_params = {
        'k': k,
        'H0': H0,
        'alpha': alpha,
        'beta': beta,
        'v0': v0,
        'gamma': gamma
    }

    # 更新参数值
    for key, value in params.items():
        if key in original_params:
            globals()[key] = value

    # 运行完整模拟流程
    events = generate_nh_poisson(lambda_func, T)
    wear_curve = simulate_wear(events)

    # 恢复原始参数
    for key, value in original_params.items():
        globals()[key] = value

    return wear_curve


# ====================== 增强泊松过程生成 ======================
def generate_nh_poisson(lambda_func, T_max):
    """优化薄化算法"""
    t = 0.0
    events = []
    max_lambda = max([lambda_func(s) for s in np.linspace(0, T_max, 1000)])

    with tqdm(total=T_max, desc="生成事件流") as pbar:
        while t < T_max:
            dt_event = np.random.exponential(1 / max_lambda)
            t += dt_event
            pbar.update(dt_event)

            if t > T_max:
                break

            if np.random.rand() < lambda_func(t) / max_lambda:
                events.append(t)

    return np.array(events)


# ====================== 动态磨损模拟引擎 ======================
def simulate_wear(event_times):
    """含环境动态变化的磨损模拟"""
    time_points = np.arange(0, T + dt, dt)
    n_counts = np.histogram(event_times, bins=time_points)[0]

    wear = np.zeros_like(time_points)
    current_wear = 0.0
    repair_idx = 0

    for i, t in enumerate(time_points):
        # 应用维修事件
        while repair_idx < len(repair_events) and t >= repair_events[repair_idx][0]:
            current_wear *= 1 - repair_events[repair_idx][1]
            repair_idx += 1

        # 获取动态环境参数
        temp, humidity = get_environment(t)
        env = env_factor(temp, humidity)

        # 计算磨损增量
        n = n_counts[i - 1]
        hardness = H(t)
        F = (alpha * n + beta) * env
        v = v0 * np.exp(-gamma * n)

        current_wear += (k / hardness) * F * v * dt
        wear[i] = current_wear

    return wear


# ====================== 增强稳定性分析模块 ======================
def stability_analysis(sim_results, n_bootstrap=1000):
    """模型稳定性验证与分析"""
    # 转换为numpy数组加速计算
    wear_matrix = np.array(sim_results)

    # 1. 时变稳定性指标
    time_stability = {
        "time": np.arange(wear_matrix.shape[1]) * dt,
        "mean": np.mean(wear_matrix, axis=0),
        "std": np.std(wear_matrix, axis=0),
        "cv": np.std(wear_matrix, axis=0) / np.mean(wear_matrix, axis=0),  # 变异系数
    }

    # 2. 自助法置信区间
    bootstrap_means = np.zeros((n_bootstrap, wear_matrix.shape[1]))
    for i in range(n_bootstrap):
        sample_indices = np.random.choice(
            wear_matrix.shape[0], size=wear_matrix.shape[0], replace=True
        )
        bootstrap_means[i] = np.mean(wear_matrix[sample_indices], axis=0)

    time_stability["ci_lower"] = np.percentile(bootstrap_means, 2.5, axis=0)
    time_stability["ci_upper"] = np.percentile(bootstrap_means, 97.5, axis=0)

    # 3. 正态性检验 (Shapiro-Wilk)
    normality = []
    for t in range(wear_matrix.shape[1]):
        stat, p = stats.shapiro(wear_matrix[:, t])
        normality.append({"time": t * dt, "W_stat": stat, "p_value": p})

    return pd.DataFrame(time_stability), pd.DataFrame(normality)


# ====================== 全局敏感性分析模块 ======================
def global_sensitivity_analysis():
    """基于Sobol方法的全局敏感性分析"""
    # 修正参数范围与默认值一致
    problem = {
        "num_vars": 5,
        "names": ["alpha", "beta", "gamma", "H0", "v0"],
        "bounds": [
            [0.05, 0.15],  # alpha (原0.1)
            [0.05, 0.15],  # beta (原0.1)
            [0.03, 0.08],  # gamma (原0.05)
            [50.0, 70.0],  # H0 (原60.0)
            [1.0, 1.5],  # v0 (原1.2)
        ],
    }

    # 生成参数样本
    param_values = saltelli.sample(problem, 512)

    # 运行模型
    outputs = []
    for params in tqdm(param_values, desc="Sobol采样进度"):
        # 将参数打包为字典
        param_dict = {
            'alpha': params[0],
            'beta': params[1],
            'gamma': params[2],
            'H0': params[3],
            'v0': params[4]
        }
        wear = run_single_simulation(**param_dict)
        outputs.append(wear[-1])  # 取最终磨损量

    # Sobol分析
    Si = sobol.analyze(problem, np.array(outputs))

    # 存储problem供可视化使用
    global sa_problem
    sa_problem = problem

    return Si


# ====================== 参数扰动分析模块 ======================
def parameter_perturb_analysis(base_params, variations=0.1, n_samples=100):
    """单参数扰动敏感性分析"""
    global base_wear  # 声明为全局变量

    # 计算基准磨损量
    print("计算基准磨损量...")
    base_wear = np.mean([run_single_simulation(**base_params)[-1] for _ in range(10)])

    results = []

    for param_name in base_params.keys():
        # 生成扰动参数
        base_value = base_params[param_name]
        perturbed_values = base_value * (
                1 + np.random.uniform(-variations, variations, n_samples)
        )

        # 运行扰动模拟
        wear_changes = []
        for val in perturbed_values:
            params = base_params.copy()
            params[param_name] = val
            wear = run_single_simulation(**params)
            wear_changes.append(wear[-1])

        # 计算敏感性指标
        sensitivity = np.std(wear_changes) / (base_params[param_name] * variations)
        results.append({
            "parameter": param_name,
            "mean_effect": np.mean(wear_changes),
            "sensitivity_index": sensitivity,
            "elasticity": (np.mean(wear_changes) - base_wear) /
                          (base_params[param_name] * variations),
        })

    return pd.DataFrame(results)


# ====================== 可视化增强模块 ======================
def plot_stability(time_stability_df, lower, upper):
    """绘制稳定性分析图"""
    plt.figure(figsize=(12, 6))

    # 均值与置信区间
    plt.plot(
        time_stability_df["time"],
        time_stability_df["mean"],
        label="Mean Wear",
        color="darkred",
    )
    # plt.fill_between(
    #     time_stability_df["time"],
    #     time_stability_df["ci_lower"],
    #     time_stability_df["ci_upper"],
    #     alpha=0.3,
    #     color="skyblue",
    #     label="95% CI",
    # )
    plt.fill_between(
        time_stability_df["time"],
        lower,
        upper,
        alpha=0.3,
        color="skyblue",
        label="90% CI",
    )

    # 变异系数次坐标
    ax2 = plt.gca().twinx()
    ax2.plot(
        time_stability_df["time"],
        time_stability_df["cv"],
        linestyle="--",
        color="green",
        label="Coefficient of Variation",
    )

    plt.title("Model Stability Analysis")
    plt.xlabel("Time (hours)")
    plt.ylabel("Accumulated Wear (Q)")
    plt.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.grid(True)
    plt.show()


def plot_sobol(Si):
    """绘制Sobol敏感性结果"""
    plt.figure(figsize=(10, 6))
    indices = pd.DataFrame({"S1": Si["S1"], "ST": Si["ST"]},
                           index=sa_problem["names"])  # 使用存储的problem

    indices.plot(kind="bar", yerr=Si["S1_conf"])
    plt.title("Sobol Sensitivity Indices")
    plt.ylabel("Sensitivity Index")
    plt.xticks(rotation=45)
    plt.show()


# ====================== 主执行流程 ======================
if __name__ == "__main__":
    # 理论计算
    print("\n计算理论磨损期望...")
    E_Q_theory = calculate_theoretical_wear()
    print(f"理论期望值: {E_Q_theory:.2f}")

    # 蒙特卡洛模拟
    print("\n运行蒙特卡洛模拟...")
    time_points = np.arange(0, T + dt, dt)
    sim_results = []

    for _ in tqdm(range(num_simulations), desc="模拟进度"):
        events = generate_nh_poisson(lambda_func, T)
        wear_curve = simulate_wear(events)
        sim_results.append(wear_curve)

    # ====================== 专业可视化 ======================
    plt.style.use(mp_seaborn_light())
    sns.set_palette("husl")
    plt.rcParams.update({"font.size": 12, "figure.dpi": 150})

    # Figure 1: Wear distribution comparison
    plt.figure(figsize=(12, 6))
    final_wear = [curve[-1] for curve in sim_results]

    sns.histplot(
        final_wear,
        kde=True,
        stat="density",
        bins=30,
        alpha=0.6,
        label="Simulation Results",
    )
    plt.axvline(
        E_Q_theory,
        color="r",
        linestyle="--",
        linewidth=2,
        label="Theoretical Expectation",
    )
    plt.axvline(
        np.mean(final_wear),
        color="g",
        linestyle=":",
        linewidth=2,
        label="Simulation Mean",
    )

    plt.xlabel("Cumulative Wear (Q)")
    plt.ylabel("Probability Density")
    plt.title("Wear Distribution: Theory vs Simulation")
    plt.legend()
    plt.tight_layout()

    # Figure 2: Time evolution trend
    plt.figure(figsize=(12, 6))
    for t_repair, _ in repair_events:
        plt.axvline(
            t_repair, color="gray", linestyle="--", alpha=0.7, label="Repair Event"
        )

    # Plot 90% confidence interval
    lower = np.percentile(sim_results, 5, axis=0)
    upper = np.percentile(sim_results, 95, axis=0)
    plt.fill_between(
        time_points,
        lower,
        upper,
        color="skyblue",
        alpha=0.3,
        label="90% Confidence Interval",
    )

    # Median trajectory
    plt.plot(
        time_points,
        np.median(sim_results, axis=0),
        color="darkred",
        linewidth=2,
        label="Median",
    )

    plt.xlabel("Time (Hours)")
    plt.ylabel("Cumulative Wear (Q)")
    plt.title("Cumulative Wear Process (with Uncertainty)")
    plt.legend()
    plt.tight_layout()

    # Figure 3: Parameter sensitivity analysis
    param_space = {
        "Temperature (°C)": np.linspace(15, 35, 50),
        "Humidity": np.linspace(0.3, 0.9, 50),
        "Gamma": np.geomspace(0.01, 0.1, 50),
    }

    plt.figure(figsize=(15, 5))
    for idx, (param_name, values) in enumerate(param_space.items()):
        plt.subplot(1, 3, idx + 1)
        results = []

        for val in tqdm(values, desc=f"Analyzing Parameter: {param_name}"):
            # Set parameter
            if param_name == "Temperature (°C)":
                global_base_temp = val
            elif param_name == "Humidity":
                global_base_humid = val
            else:
                global_gamma = val

            # Run simulation 10 times and take the average
            wear_values = []
            for _ in range(10):
                events = generate_nh_poisson(lambda_func, T)
                wear_curve = simulate_wear(events)
                wear_values.append(wear_curve[-1])

            results.append(np.mean(wear_values))

        # Nonlinear regression analysis
        sns.regplot(
            x=values,
            y=results,
            order=2,
            ci=95,
            scatter_kws={"alpha": 0.3},
            line_kws={"color": "red"},
        )
        plt.xlabel(param_name)
        plt.ylabel("Expected Wear (Q)")
        plt.title(f"{param_name} Sensitivity (Quadratic Fit)")

    plt.tight_layout()
    plt.show()

    # 新增分析模块
    print("\n正在进行稳定性分析...")
    time_stability_df, normality_df = stability_analysis(sim_results)

    print("\n进行全局敏感性分析...")
    Si = global_sensitivity_analysis()

    print("\n进行参数扰动分析...")
    base_params = {"alpha": 0.5, "beta": 0.1, "gamma": 0.05, "H0": 10.0, "v0": 1.2}
    perturb_df = parameter_perturb_analysis(base_params)

    # 生成可视化报告
    plot_stability(time_stability_df, lower, upper)
    plot_sobol(Si)

    # 保存统计结果
    time_stability_df.to_csv("time_stability.csv", index=False)
    normality_df.to_csv("normality_test.csv", index=False)
    perturb_df.to_csv("parameter_perturbation.csv", index=False)



    # 输出关键指标
    # print("\n=== 关键稳定性指标 ===")
    # print(f"最终时刻变异系数(CV): {time_stability_df['cv'].iloc[-1]:.4f}")
    # print(
    #     f"平均95% CI宽度: {np.mean(time_stability_df['ci_upper'] - time_stability_df['ci_lower']):.4f}"
    # )
    #
    # print("\n=== Sobol全局敏感性指数 ===")
    # perturb_df = parameter_perturb_analysis(base_params)
    # print(Si)
    #
    # print("\n=== 参数扰动分析结果 ===")
    # print(perturb_df)
