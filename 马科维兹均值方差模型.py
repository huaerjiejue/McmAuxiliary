import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import scipy.stats as scs
import statsmodels.api as sm
import scipy.optimize as sco
import scipy.interpolate as sci
from tqdm import tqdm  # 看进度条
from MathModels.Plot.styles import mp_seaborn_light
from MathModels.Plot.colors import get_inferno, get_plasma

warnings.filterwarnings("ignore")
plt.style.use(mp_seaborn_light())

# 读取数据
df = pd.read_excel("data.xlsx", index_col=0)
hist_close = pd.pivot_table(df, index="index", columns="code", values="close")
returns = np.log(hist_close / hist_close.shift(1))  # 计算收益率
returns_clean = returns.dropna(axis=0)

# 提前计算均值和协方差
mean_returns = returns_clean.mean() * 252
cov_matrix = returns_clean.cov() * 252

# 生成随机权重并计算组合收益率和方差
noa = returns_clean.shape[1]  # 资产个数
n_portfolios = 15000
weights = np.random.random((n_portfolios, noa))
weights /= weights.sum(axis=1, keepdims=True)

port_returns = np.dot(weights, mean_returns)
port_variance = np.array([np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))) for w in weights])

# 画图
# plt.figure(figsize=(8, 4))
# plt.scatter(
#     port_variance, port_returns, c=port_returns / port_variance, cmap="Reds", marker="o"
# )
# plt.grid(True)
# plt.xlabel("expected volatility")
# plt.ylabel("expected return")
# plt.colorbar(label="Sharpe ratio")
# plt.show()


# 定义统计函数
def statistics(weights):
    weights = np.array(weights)
    port_returns = np.sum(mean_returns * weights)
    port_variance = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return np.array([port_returns, port_variance, port_returns / port_variance])


# 最小化夏普比率
def min_sharpe(weights):
    return -statistics(weights)[2]


# 最小化方差
def min_variance(weights):
    return statistics(weights)[1] ** 2


# 约束条件：权重之和等于1
cons = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
bnds = tuple((0, 1) for x in range(noa))

# 优化夏普比率
opts = sco.minimize(
    min_sharpe, noa * [1.0 / noa], method="SLSQP", bounds=bnds, constraints=cons
)

# 优化方差
optv = sco.minimize(
    min_variance, noa * [1.0 / noa], method="SLSQP", bounds=bnds, constraints=cons
)


# 计算有效前沿
def min_port(weights):
    return statistics(weights)[1]


trets = np.linspace(port_returns.min(), port_returns.max(), 200)  # 收益率
tvols = []

for tret in tqdm(trets):  # 给定收益率，最小标准差
    cons = (
        {"type": "eq", "fun": lambda x: statistics(x)[0] - tret},
        {"type": "eq", "fun": lambda x: np.sum(x) - 1},
    )
    res = sco.minimize(
        min_port, noa * [1.0 / noa], method="SLSQP", bounds=bnds, constraints=cons
    )
    tvols.append(res["fun"])
tvols = np.array(tvols)

# 计算CML
ind = np.argmin(tvols)  # 选择有效前沿上部分的数据
evols = tvols[ind:]
erets = trets[ind:]
tck = sci.splrep(evols, erets, k=3)  # spline差值


def equations(p, rf=0.04):
    eq1 = rf - p[0]  # 截距等于rf
    eq2 = rf + p[1] * p[2] - sci.splev(p[2], tck, der=0)  # 曲线和CML相切
    eq3 = p[1] - sci.splev(p[2], tck, der=1)  # 曲线导数等于CML的斜率
    return eq1, eq2, eq3


opt = sco.fsolve(equations, [0.01, 0.5, 0.15])  # p[0] p[1] p[2]初始值

plt.figure(figsize=(10, 6))

# 使用渐变色表示夏普比率
scatter = plt.scatter(
    port_variance,
    port_returns,
    c=port_returns / port_variance,  # 夏普比率
    cmap="viridis",  # 使用 viridis 颜色映射
    marker="o",  # 圆形标记
    s=50,  # 标记大小
    alpha=0.8,  # 透明度
    edgecolors="w",  # 标记边缘颜色
    linewidths=0.5,  # 边缘线宽
)

# 添加颜色条
cbar = plt.colorbar(scatter)
cbar.set_label("Sharpe Ratio", fontsize=12)

# 绘制有效前沿
plt.plot(tvols, trets, color="darkorange", linewidth=3, label="Efficient Frontier")

# 绘制 CML
cx = np.linspace(port_variance.min() - 0.05, port_variance.max() + 0.05)
plt.plot(
    cx,
    opt[0] + opt[1] * cx,
    color="blue",
    linewidth=2,
    linestyle="--",
    label="Capital Market Line",
)

# 绘制切点
plt.plot(
    opt[2],
    sci.splev(opt[2], tck, der=0),
    "r*",
    markersize=15,
    label="Tangency Portfolio",
)

# 添加网格
plt.grid(True, linestyle="--", alpha=0.6)

# 设置标题和标签
plt.title("Efficient Frontier with Capital Market Line", fontsize=16)
plt.xlabel("Expected Volatility", fontsize=12)
plt.ylabel("Expected Return", fontsize=12)

# 设置背景颜色
plt.gca().set_facecolor("#f7f7f7")

# 显示图例
plt.legend(fontsize=12)

# 美化布局
plt.tight_layout()

# 显示图表
plt.show()

# 计算最优组合
cons = (
    {"type": "eq", "fun": lambda x: statistics(x)[0] - sci.splev(opt[2], tck, der=0)},
    {"type": "eq", "fun": lambda x: np.sum(x) - 1},
)
res = sco.minimize(
    min_port, noa * [1.0 / noa], method="SLSQP", bounds=bnds, constraints=cons
)
print(res["x"])
print(statistics(res["x"]))
