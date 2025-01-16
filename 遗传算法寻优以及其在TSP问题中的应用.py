import numpy as np
from sko.GA import GA, GA_TSP
import pandas as pd
from scipy import spatial
import matplotlib.pyplot as plt
from matplotlib import cm


def schaffer(p):  # 取最小值！！！
    """This function has plenty of local minimum, with strong shocks
    global minimum at (0,0) with value 0"""
    x1, x2 = p
    x = np.square(x1) + np.square(x2)
    return 0.5 + (np.square(np.sin(x)) - 0.5) / np.square(1 + 0.001 * x)


ga = GA(
    func=schaffer,
    n_dim=2,
    size_pop=50,
    max_iter=800,
    prob_mut=0.001,
    lb=[-1, -1],
    ub=[1, 1],
    precision=1e-7,
)
# best_x, best_y = ga.run()
# print("best_x:", best_x, "\n", "best_y:", best_y)


def plot_run(ga):
    # 提取适应度历史数据
    Y_history = pd.DataFrame(ga.all_history_Y)

    # 创建画布
    fig, ax = plt.subplots(2, 1)

    # 绘制每一代所有个体的适应度值
    ax[0].plot(Y_history.index, Y_history.values, ".", color="red")
    ax[0].set_title("Fitness Values of All Individuals in Each Generation")
    ax[0].set_xlabel("Generation")
    ax[0].set_ylabel("Fitness Value")

    # 绘制每一代最优适应度值的变化趋势
    Y_history.min(axis=1).cummin().plot(kind="line", ax=ax[1])
    ax[1].set_title("Best Fitness Value Over Generations")
    ax[1].set_xlabel("Generation")
    ax[1].set_ylabel("Best Fitness Value")

    # 调整布局并显示图形
    plt.tight_layout()
    plt.show()


# plot_run(ga)


def plot_3d():
    # 定义 x1 和 x2 的范围
    X_BOUND = [-10, 10]  # x1 的取值范围
    Y_BOUND = [-10, 10]  # x2 的取值范围

    # 生成网格数据
    X = np.linspace(*X_BOUND, 100)
    Y = np.linspace(*Y_BOUND, 100)
    X, Y = np.meshgrid(X, Y)
    Z = np.array([schaffer([x, y]) for x, y in zip(X.ravel(), Y.ravel())]).reshape(
        X.shape
    )

    # 创建三维图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # 绘制三维曲面
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x1, x2)")
    ax.set_title("3D Surface Plot of Schaffer Function")

    # 添加颜色条
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # 显示图形
    plt.show()


# plot_3d()
def plot_contour():
    # 定义 x1 和 x2 的范围
    X_BOUND = [-10, 10]  # x1 的取值范围
    Y_BOUND = [-10, 10]  # x2 的取值范围

    # 生成网格数据
    X = np.linspace(*X_BOUND, 100)
    Y = np.linspace(*Y_BOUND, 100)
    X, Y = np.meshgrid(X, Y)
    Z = np.array([schaffer([x, y]) for x, y in zip(X.ravel(), Y.ravel())]).reshape(
        X.shape
    )

    # 创建等高线图
    plt.figure()
    contour = plt.contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.colorbar(contour)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Contour Plot of Schaffer Function")

    # 显示图形
    plt.show()


# 调用绘图函数
# plot_contour()

# 利用遗传算法解 TSP 问题
num_points = 50
points_coordinate = np.random.rand(num_points, 2)  # generate coordinate of points
distance_matrix = spatial.distance.cdist(
    points_coordinate, points_coordinate, metric="euclidean"
)


def cal_total_distance(routine):
    """
    The objective function. input routine, return total distance.
    cal_total_distance(np.arange(num_points))
    """
    (num_points,) = routine.shape
    return sum(
        [
            distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]]
            for i in range(num_points)
        ]
    )


ga_tsp = GA_TSP(
    func=cal_total_distance, n_dim=num_points, size_pop=50, max_iter=500, prob_mut=1
)
best_points, best_distance = ga_tsp.run()
fig, ax = plt.subplots(1, 2)
best_points_ = np.concatenate([best_points, [best_points[0]]])
best_points_coordinate = points_coordinate[best_points_, :]
ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], "o-r")
ax[1].plot(ga_tsp.generation_best_Y)
plt.show()
print(f"best_points: {best_points}\nbest_distance:{best_distance}")
