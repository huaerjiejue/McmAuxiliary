##这里主要是numpy，sympy，scipy的使用，对于简单矩阵的求解之类的
import numpy as np
from sympy import symbols, solve, nonlinsolve
from scipy.optimize import fsolve
from math import sin, cos, pi

# 坐标旋转
theta = np.radians(30) # 30度转换为弧度

rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

# 旋转前的坐标
point = np.array([1, 0])
rotated_point = np.dot(rotation_matrix, point)

print(f"旋转前的坐标: {point}")
print(f"旋转后的坐标: {rotated_point}")

# 三维空间旋转
alpha = np.radians(30)  # Z轴旋转30度
beta = np.radians(45)  # Y轴旋转45度
gamma = np.radians(60)  # X轴旋转60度

R_z = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                [np.sin(alpha), np.cos(alpha), 0],
                [0, 0, 1]])
R_y = np.array([[np.cos(beta), 0, np.sin(beta)],
                [0, 1, 0],
                [-np.sin(beta), 0, np.cos(beta)]])
R_x = np.array([[1, 0, 0],
                [0, np.cos(gamma), -np.sin(gamma)],
                [0, np.sin(gamma), np.cos(gamma)]])
R = np.dot(R_x, np.dot(R_y, R_z))
print(f"三维空间旋转矩阵: {R}")

# np中linalg模块中的函数
matrix = np.array([[1, 2], [3, 4]])
# 求解逆矩阵
matrix_inv = np.linalg.inv(matrix)
print(f"矩阵A的逆矩阵: {matrix_inv}")

# 特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(matrix)
print(f"矩阵A的特征值: {eigenvalues}")
print(f"矩阵A的特征向量: {eigenvectors}")

# 矩阵的行列式
det = np.linalg.det(matrix)
print(f"矩阵A的行列式: {det}")

# 矩阵的秩
rank = np.linalg.matrix_rank(matrix)
print(f"矩阵A的秩: {rank}")

# 矩阵的迹
trace = np.trace(matrix)
print(f"矩阵A的迹: {trace}")

# 矩阵的QR分解
Q, R = np.linalg.qr(matrix)
print(f"矩阵A的QR分解: Q={Q}, R={R}")

# 矩阵的SVD分解
U, S, V = np.linalg.svd(matrix)
print(f"矩阵A的SVD分解: U={U}, S={S}, V={V}")

# Sympy求解方程组
x, y = symbols('x y')
eq1 = 2 * x + 3 * y - 6
eq2 = 3 * x + 2 * y - 12
solution = solve((eq1, eq2), (x, y))
print(f"线性方程组的解: {solution}")
a, b, c, d = symbols('a b c d')
# print(nonlinsolve([a * b - 1, a - c], [a, b]))
print(f"非线性方程组的解：{nonlinsolve([a * b -1, a-c], [a, b])}")

# 当非线性方程组没有确定解的时候，考虑使用scipy的fsolve进行求解
def equations(vars):
    x, y, theta = vars
    L1, L2, L3 = 3, 3, 3
    p1, p2, p3 = 5, 5, 3
    x1, x2, y2 = 5, 0, 6
    # 根据问题描述定义的方程
    eq1 = (x + L3*cos(theta) - x1)**2 + (y + L3*sin(theta))**2 - p2**2
    eq2 = x**2 + y**2 - p1**2
    eq3 = (x + L2*cos(pi/3 + theta))**2 + (y + L2*sin(pi/3 + theta) - y2)**2 - p3**2
    return [eq1, eq2, eq3]
# 初始猜测值
initial_guess = [-1.37, 4.80, 0.12]
# 使用fsolve求解方程组
result = fsolve(equations, initial_guess)
print(f"Scipy使用fsolve进行非线性求解：{result}")
