from sko.SA import SAFast, SA_TSP
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
import cmath#引入可以计算复数的函数模块

#定义待求解的函数
def func(x):
    x1, x2, x3 = x
    return x1*complex(0,1)*cmath.sin(x2)+x3#complex(0,1)为一个虚数单位

#实例化算法，并加入初始解x0=[1, 1, 1]
# sa = SAFast(func=func, x0=[1, 1, 1])
sa_fast = SAFast(func=func, x0=[1, 1, 1], T_max=1, T_min=1e-9, q=0.99, L=300, max_stay_counter=150,
                 lb=[-1, 1, -1], ub=[2, 3, 4])

#进行拟合
x_star, y_star = sa_fast.fit()

#生成最优解x和最优值y
print(x_star, y_star)

# TSP问题
num_points = 50
points_coordinate = np.random.rand(num_points, 2)  # generate coordinate of points
distance_matrix = spatial.distance.cdist(
    points_coordinate, points_coordinate, metric="euclidean"
)


def cal_total_distance(routine):
    """The objective function. input routine, return total distance.
    cal_total_distance(np.arange(num_points))"""
    (num_points,) = routine.shape
    return sum(
        [
            distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]]
            for i in range(num_points)
        ]
    )

sa_tsp = SA_TSP(func=cal_total_distance, x0=range(num_points), T_max=800, T_min=1, L=1000)
best_points, best_distance = sa_tsp.run()
print(best_points, best_distance, cal_total_distance(best_points))
fig, ax = plt.subplots(1, 2)
best_points_ = np.concatenate([best_points, [best_points[0]]])
best_points_coordinate = points_coordinate[best_points_, :]
ax[0].plot(sa_tsp.best_y_history)
ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("Distance")
ax[1].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], marker='o', markerfacecolor='b', color='c', linestyle='-')
ax[1].set_xlabel("Longitude")
ax[1].set_ylabel("Latitude")
plt.show()