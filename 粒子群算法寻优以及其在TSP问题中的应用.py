from sko.PSO import PSO, PSO_TSP
import matplotlib.pyplot as plt
from scipy import spatial
import numpy as np


def demo_func(x):
    x1, x2, x3 = x
    return x1**2 + (x2 - 0.05) ** 2 + x3**2


pso = PSO(
    func=demo_func,
    dim=3,
    pop=40,
    max_iter=150,
    lb=[0, -1, 0.5],
    ub=[1, 1, 1],
    w=0.8,
    c1=0.5,
    c2=0.5,
)
pso.run()
print("best_x is ", pso.gbest_x, "best_y is", pso.gbest_y)
# plt.plot(pso.gbest_y_hist)
# plt.show()

# 使用粒子群算法解决TSP问题
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


pso_tsp = PSO_TSP(
    func=cal_total_distance,
    n_dim=num_points,
    size_pop=200,
    max_iter=800,
    w=0.8,
    c1=0.1,
    c2=0.1,
)
best_points, best_distance = pso_tsp.run()
print("best_distance", best_distance)
fig, ax = plt.subplots(1, 2)
best_points_ = np.concatenate([best_points, [best_points[0]]])
best_points_coordinate = points_coordinate[best_points_, :]
ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], "o-r")
ax[1].plot(pso_tsp.gbest_y_hist)
plt.show()
