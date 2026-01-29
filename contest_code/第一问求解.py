# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt
#
# # 输入数据
# data = {
#     "负载(N)": [100, 200, 300, 400, 500],
#     "M1": [9.43, 17.79, 27.49, 33.56, 40.33],
#     "M2": [13.73, 23.5, 32.27, 40.05, 45.68],
#     "M3": [12.81, 19.08, 26.77, 37.89, 40.29],
#     "L1": [7.18, 12.75, 15.42, 18.5, 21.54],
#     "L2": [6.06, 11.2, 17.23, 20.1, 22.75],
#     "L3": [6.42, 10.15, 18.66, 22.39, 25.33],
#     "T1": [16.16, 30.92, 37.46, 49.43, 55.88],
#     "T2": [11.2, 17.85, 24.19, 29.3, 34.62],
#     "T3": [16, 22.97, 30.98, 39, 42.82]
# }
#
# df = pd.DataFrame(data)
#
# # 设置预测负载
# predict_load = 600
#
# # 准备存储结果
# results = []
# coefficients = {}
#
# # 创建绘图
# plt.figure(figsize=(15, 20))
# plt.suptitle("Linear Regression Analysis", y=1.02, fontsize=14)
#
# # 对每个物理量进行回归分析
# for idx, column in enumerate(df.columns[1:]):  # 跳过第一列（负载）
#     # 准备数据
#     X = df[["负载(N)"]].values
#     y = df[column].values
#
#     # 训练模型
#     model = LinearRegression()
#     model.fit(X, y)
#
#     # 存储系数
#     coefficients[column] = {
#         "slope": model.coef_[0],
#         "intercept": model.intercept_,
#         "r_squared": model.score(X, y)
#     }
#
#     # 预测600N
#     prediction = model.predict([[predict_load]])[0]
#     results.append((column, prediction))
#
#     # 绘制子图
#     plt.subplot(3, 3, idx + 1)
#     plt.scatter(X, y, color='blue', label='Actual Data')
#     plt.plot(X, model.predict(X), color='red', label='Fitted Line')
#     plt.xlabel('Load (N)')
#     plt.ylabel(column)
#     plt.title(f"{column} Regression\n"
#               f"y = {model.coef_[0]:.4f}x + {model.intercept_:.2f}\n"
#               f"R² = {model.score(X, y):.3f}")
#     plt.grid(True)
#     plt.legend()
#
# plt.tight_layout()
# plt.show()
#
# # 显示预测结果
# print("\n预测结果 (600N 负载):")
# print("+------+-----------+")
# print("| 参数 | 预测值    |")
# print("+------+-----------+")
# for param, value in results:
#     print(f"| {param:4} | {value:7.1f}   |")
# print("+------+-----------+")

# 显示回归方程（可选）
# print("\n回归方程:")
# for param, vals in coefficients.items():
#     print(f"{param}: y = {vals['slope']:.4f}x + {vals['intercept']:.2f} (R²={vals['r_squared']:.3f})")

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# # 输入数据（假设负载与之前结构相同：100, 200, 300, 400, 500 牛）
# data = {
#     "负载(N)": [100, 200, 300, 400, 500],
#     "G1": [1.25, 2.72, 4.4, 4.54, 4.29],
#     "G2": [1.39, 2.65, 4.3, 4.44, 3.81],
#     "G3": [2.68, 5.33, 7.03, 7.64, 7.31]
# }
#
# df = pd.DataFrame(data)
# predict_load = 600  # 需要预测的负载
# degree = 2  # 多项式阶数（可修改为 2/3/4）
#
# # 准备存储结果
# results = []
# models = {}
#
# # 创建绘图
# plt.figure(figsize=(15, 5))
# plt.suptitle(f"{degree}次多项式回归分析", y=1.05, fontsize=14)
#
# # 对每个G参数进行多项式回归
# for idx, column in enumerate(["G1", "G2", "G3"], 1):
#     X = df[["负载(N)"]].values
#     y = df[column].values
#
#     # 生成多项式特征
#     poly = PolynomialFeatures(degree=degree, include_bias=False)
#     X_poly = poly.fit_transform(X)
#
#     # 训练模型
#     model = LinearRegression()
#     model.fit(X_poly, y)
#
#     # 存储模型
#     models[column] = {
#         "model": model,
#         "poly": poly,
#         "r2": r2_score(y, model.predict(X_poly))
#     }
#
#     # 预测600N
#     X_pred = poly.transform([[predict_load]])
#     prediction = model.predict(X_pred)[0]
#     results.append((column, prediction))
#
#     # 绘制子图
#     plt.subplot(1, 3, idx)
#     x_plot = np.linspace(100, 600, 100).reshape(-1, 1)
#     x_plot_poly = poly.transform(x_plot)
#
#     plt.scatter(X, y, color='blue', label='实际数据')
#     plt.plot(x_plot, model.predict(x_plot_poly), 'r-', label='拟合曲线')
#     plt.scatter([predict_load], [prediction], color='green', marker='*', s=100, label='预测点')
#
#     plt.title(f"{column}\nR² = {models[column]['r2']:.3f}")
#     plt.xlabel('负载 (N)')
#     plt.ylabel(column)
#     plt.grid(True)
#     plt.legend()
#
# plt.tight_layout()
# plt.show()
#
# # 显示预测结果
# print(f"\n预测结果 ({predict_load}N 负载, {degree}次多项式):")
# print("+------+-----------+")
# print("| 参数 | 预测值    |")
# print("+------+-----------+")
# for param, value in results:
#     print(f"| {param:4} | {value:7.2f}   |")
# print("+------+-----------+")
#
# # 显示多项式方程（示例显示G1的方程）
# column = "G1"
# coefs = models[column]["model"].coef_
# intercept = models[column]["model"].intercept_
# equation = f"y = {intercept:.2f} "
# for d in range(1, degree + 1):
#     equation += f"+ {coefs[d - 1]:.4f}x^{d} "
# print(f"\n示例方程 ({column}):\n{equation}")


import numpy as np
from scipy.stats import truncnorm

# ====================== 初始化参数 ======================
STAIR_ROWS = 20  # 石阶总行数
STAIR_COLS = 50  # 石阶总列数
FOOT_LENGTH = 6  # 脚掌覆盖列数

# 初始化上行/下行压力矩阵（扩展为20行）
upstirs = np.zeros([STAIR_ROWS, FOOT_LENGTH])
downstirs = np.zeros([STAIR_ROWS, FOOT_LENGTH])

# 填充上行压力数据（示例模式）
upstirs[0:3, 0:2] = 45.3  # 前3行，前2列为高压力区
upstirs[0:3, 2:6] = 28.4  # 前3行，后4列为中压力区
upstirs[3:7, 0:2] = 50.6  # 中间行模式...
upstirs[3:7, 2:4] = 50.3
upstirs[3:7, 4:6] = 48.5
upstirs[7:11] = 41.2  # 均匀分布区
upstirs[11:15] = 34.6  # 后部低压力区
upstirs[15:20] = 25.0  # 新增行填充

# 填充下行压力数据（示例模式）
downstirs[0:4] = 21.7
downstirs[4:8] = 29.7
downstirs[8:12, 0:2] = 47.4
downstirs[8:12, 2:4] = 53.9
downstirs[8:12, 4:6] = 57
downstirs[12:15, 0:4] = 28.5
downstirs[12:15, 4:6] = 43.8
downstirs[15:20] = 30.0  # 新增行填充

# 初始化石阶压力累积矩阵
stone_stirs = np.zeros([STAIR_ROWS, STAIR_COLS])


# ====================== 核心函数 ======================
def get_foot_position(x):
    """根据行号x计算脚尖位置偏移量"""
    # 假设：x=0为最前端，x=19为最后端
    # 脚尖距离前端边界的比例（0.0~1.0）
    front_ratio = x / STAIR_ROWS
    # 动态调整起始列：前端行更靠近起始列
    base_y = int(STAIR_COLS * front_ratio * 0.6)
    return base_y


def safe_add_pressure(target, x, y_start, pressure):
    """安全地将压力数据添加到石阶矩阵"""
    y_end = y_start + FOOT_LENGTH
    if y_end > STAIR_COLS:
        y_end = STAIR_COLS
    cols_to_fill = y_end - y_start
    target[x, y_start:y_end] += pressure[:cols_to_fill]


# ====================== 模拟过程 ======================
for _ in range(1000):
    # 随机选择上下行（60%上行，40%下行）
    is_up = np.random.rand() < 0.8

    # 生成行号x（集中在中间区域）
    x = truncnorm.rvs(
        a=(0 - 0.5 - 10) / 3,
        b=(STAIR_ROWS - 1 + 0.5 - 10) / 3,
        loc=10,
        scale=3
    ).astype(int)

    # 根据x计算基础起始列
    base_y = get_foot_position(x)

    # 生成随机偏移（-2到+2列）
    y_offset = np.random.randint(-2, 3)
    y_start = base_y + y_offset

    # 确保不越界
    y_start = max(0, min(y_start, STAIR_COLS - FOOT_LENGTH))

    # 获取压力数据
    if is_up:
        pressure_data = upstirs[x]
    else:
        pressure_data = downstirs[x]

    # 累加到石阶矩阵
    safe_add_pressure(stone_stirs, x, y_start, pressure_data)

# ====================== 可视化 ======================
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.imshow(stone_stirs, cmap='hot', aspect='auto')
plt.colorbar(label='Pressure (kPa)')
plt.title('Cumulative Pressure Distribution on Stairs')
plt.xlabel('Column Position')
plt.ylabel('Row Position')
plt.show()