"""
- 线性规划的基本模型与求解
- 非线性规划的基本模型与求解
- 整数规划的基本模型与求解
- 动态规划的基本模型与求解
- 多目标规划的基本模型与求解
"""
import numpy as np
from scipy.optimize import linprog, minimize
from sko.GA import GA

"""
minimize: z=3*14x_1+4*16x_2
subject to:
    12x_1+x_2<=50
    x_1+x_2<=50
    3x_1<=100
    x_1,x_2>=0
"""
c = np.array([-3*14, -4*16])
A = np.array([[12, 1], [1, 1], [3, 0]])
b = np.array([50, 50, 100])
bounds = [(0, None), (0, None)]
res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
print(res)

# 非线性规划
def fun(x):
    return 10.5 + 0.3*x[0] + 0.32*x[1] + 0.32*x[2] + 0.0007*x[0]**2 + 0.0004*x[1]**2 + 0.00045*x[2]**2

cons = ({
    'type': 'eq',
    'fun': lambda x: x[0]+x[1]+x[2]-700
})
b1, b2, b3 = (100, 200), (120, 250), (150, 300)
x_0 = np.array([100, 120, 150])
res = minimize(fun, x_0, constraints=cons, bounds=[b1, b2, b3], method='SLSQP')
print(res)
ga = GA(func=fun, n_dim=3, size_pop=500, max_iter=500, lb=[100, 120, 150],
        ub=[200, 250, 300], constraint_eq=[lambda x: 700-x[0]-x[1]-x[2]])
# best_x, best_y = ga.run()
# print(best_x, best_y)

"""
f(x) = (x0-2)^2 + (x1-3)^2 + (x2-4)^2
subject to:
    x0+x1+x2=10
    x0,x1,x2>=0
    x0,x1,x2<=10
    x0^2+x1^2<=25
    x1+x2>=6
下面使用两个算法，一个是scipy.optimize.minimize，另一个是遗传算法GA
"""
def fun(x):
    return (x[0]-2)**2 + (x[1]-3)**2 + (x[2]-4)**2
cons = [
    {'type': 'eq', 'fun': lambda x: x[0]+x[1]+x[2]-10},
    {'type': 'ineq', 'fun': lambda x: 25-x[0]**2-x[1]**2},
    {'type': 'ineq', 'fun': lambda x: x[1]+x[2]-6}
]
b1, b2, b3 = (0, 10), (0, 10), (0, 10)
x_0 = np.array([0, 0, 0])
res = minimize(fun, x_0, constraints=cons, bounds=[b1, b2, b3], method='SLSQP')
print("Minimize Results:")
print(f"Optimal value: {res.fun}")
print(f"Optimal variables: {res.x}")
cons_eq = [lambda x: 10-x[0]-x[1]-x[2]]
cons_ueq = [lambda x: 25-x[0]**2-x[1]**2, lambda x: x[1]+x[2]-6]
ga = GA(func=fun, n_dim=3, size_pop=500, max_iter=500, lb=[0, 0, 0], ub=[10, 10, 10],
        constraint_eq=cons_eq, constraint_ueq=cons_ueq)
best_x, best_y = ga.run()
print("GA Results:")
print(f"Optimal value: {best_y}")
print(f"Optimal variables: {best_x}")

