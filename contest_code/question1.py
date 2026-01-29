import math

from matplotlib.cm import ScalarMappable

from data.hardness import rock_hardness
from data.wear_coefficient import rock_dry
import matplotlib.pyplot as plt
import numpy as np

def volume_wear_rate(k, F, v, H):
    '''
    计算体积磨损率
    :param k:磨损系数
    :param F:法向力
    :param v：滑动速度
    :param H:材料硬度
    :return:
    '''
    return k*F*v/H

def depth_wear_rate(k, F, v, H, A):
    '''
    计算深度磨损率
    :param k:磨损系数
    :param F:法向力
    :param v：滑动速度
    :param H:材料硬度
    :param A:接触面积
    :return:
    '''
    return k*F*v/(H*A)


def alpha_theta(theta, theta0, sigma):
    """
    计算方向权重因子 alpha(theta)。

    参数:
    theta (float): 当前人流方向与参考方向的夹角（单位：度）。
    theta0 (float): 主导方向（如楼梯中心线，单位：度）。
    sigma (float): 方向分布宽度参数。

    返回:
    float: 方向权重因子 alpha(theta)。
    """
    return math.exp(-((theta - theta0) ** 2) / (2 * sigma ** 2))


def directional_wear_correction(V_dot, theta, theta0, sigma):
    """
    计算方向性磨损修正后的磨损率。

    参数:
    V_dot (float): 原始磨损率。
    theta (float): 当前人流方向与参考方向的夹角（单位：度）。
    theta0 (float): 主导方向（单位：度）。
    sigma (float): 方向分布宽度参数。

    返回:
    float: 修正后的磨损率 V_dot_dir。
    """
    alpha = alpha_theta(theta, theta0, sigma)
    return alpha * V_dot


# 不同硬度，不同不同摩擦系数下的磨损率
v = 1
F = 600
vwr = {}

hardness_range = np.arange(40, 110, 10)  # 40-100，步长10
k_range = np.arange(0.225, 0.51, 0.05)  # 摩擦系数范围

# 生成数据字典
for H in hardness_range:
    for k in k_range:
        vwr[(H, k)] = volume_wear_rate(round(k,3), F, v, H)  # 保留3位小数避免浮点误差

# 转换为二维数组用于绘图
hardness_values = sorted(list(set(h for h, _ in vwr.keys())))
k_values = sorted(list(set(k for _, k in vwr.keys())))
wear_matrix = np.zeros((len(hardness_values), len(k_values)))

for i, H in enumerate(hardness_values):
    for j, k in enumerate(k_values):
        wear_matrix[i, j] = vwr[(H, k)]

# 创建绘图
plt.figure(figsize=(10, 6))
cmap = plt.get_cmap('viridis')  # 选择颜色映射
norm = plt.Normalize(min(hardness_values), max(hardness_values))  # 颜色标准化

# 绘制每条硬度曲线
for idx, H in enumerate(hardness_values):
    color = cmap(norm(H))
    plt.plot(k_values,
             wear_matrix[idx, :],
             marker='o',
             linestyle='-',
             color=color,
             label=f'H={H}')

# 添加颜色条
# sm = ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# cbar = plt.colorbar(sm)
# cbar.set_label('Rock Hardness (H)')

# 图表装饰
plt.title("Wear Rate vs Friction Coefficient (F=600N, v=1m/s)")
plt.xlabel("Friction Coefficient (k)")
plt.ylabel("Volume Wear Rate")
plt.grid(True, alpha=0.3)
plt.xticks(k_values)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 将图例移到右侧
plt.tight_layout()
plt.show()

#
V0 = 3
# 设置均值和标准差
mu = 0
sigma = 1

thetas = np.linspace(-np.pi/4, np.pi/4, 1000)  # 生成一组角度
V = [directional_wear_correction(V0, theta, mu, sigma) for theta in thetas]  # 计算方向性磨损修正后的磨损率

# # 绘制图表
# plt.figure(figsize=(10, 6))
# plt.plot(thetas, V, label='Directional Wear Correction')
# plt.axvline(mu, color='r', linestyle='--', label='Main Direction')
# plt.fill_between(thetas, V, color='skyblue', alpha=0.4)
# plt.title('Directional Wear Correction vs Angle')
# plt.xlabel('Angle (radians)')
# plt.ylabel('Wear Rate Correction')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

def hardness_with_temperature(A, B, T, T0):
    # B 0.01~0.005
    # A 初始硬度
    return A*math.exp(-B * (T - T0))

b_low = 0.005
b_high = 0.01
A = 40
T = np.arange(0, 500, 0.1)
t0 = 30
H_low = [hardness_with_temperature(A, b_low, t, t0) for t in T]
H_high = [hardness_with_temperature(A, b_high, t, t0) for t in T]

plt.figure(figsize=(10, 6))
plt.plot(T, H_low, label='Hardness vs Temperature', color='r')
plt.plot(T, H_high, label='Hardness vs Temperature', color='b')
plt.title('Hardness vs Temperature')
plt.xlabel('Temperature (°C)')
plt.ylabel('Hardness')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()