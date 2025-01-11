import numpy as np
from scipy.integrate import quad, odeint, solve_ivp
from sympy import *
from sympy.solvers.pde import pdsolve
import matplotlib.pyplot as plt


# 计算定积分
def f(x):
    return np.cos(2 * np.pi * x) * np.exp(-x) + 1.2


integral, error = quad(f, 0, 5)
print(f"定积分的值: {integral}")
print(f"误差的值: {error}")

# 计算微分
x = np.linspace(0, 10, 100)
y = x**2
dydx = np.gradient(y, x)
# 在x=5处的导数
print(f"在x=5处的导数: {dydx[np.argmin(np.abs(x - 5))]}")

# 使用sympy计算微分方程
y = symbols("y", cls=Function)
x = symbols("x")
eq = Eq(y(x).diff(x, 2) + 2 * y(x).diff(x) + y(x), sin(x))  # y'' + 2y' + y = sin(x)
sol = dsolve(eq)
print(f"微分方程的解: {sol}")

t = symbols("t")
x1, x2, x3 = symbols("x1,x2,x3", cls=Function)
"""
x'1=2x1-3x2+3x3, x1(0)=1
x'2=4x1-5x2+3x3, x2(0)=2
x'3=4x1-4x2+2x3, x3(0)=3
"""
eq = [
    x1(t).diff(t) - 2 * x1(t) + 3 * x2(t) - 3 * x3(t),
    x2(t).diff(t) - 4 * x1(t) + 5 * x2(t) - 3 * x3(t),
    x3(t).diff(t) - 4 * x1(t) + 4 * x2(t) - 2 * x3(t),
]
con = {x1(0): 1, x2(0): 2, x3(0): 3}
s = dsolve(eq, ics=con)
print(f"微分方程组的解: {s}")
# 第二种线性微分方程组的解法，使用矩阵
A = Matrix([[2, -3, 3], [4, -5, 3], [4, -4, 2]])
X = Matrix([x1(t), x2(t), x3(t)])
eq = X.diff(t) - A * X
sol = dsolve(eq, ics=con)
print(f"微分方程组的解: {sol}")

# sympy绘图
t = symbols("t")
# plot(sin(t), (t, -2*pi, 2*pi), line_color='r', title='sin(t)')
# plot(cos(t), (t, -2*pi, 2*pi), line_color='b', title='cos(t)')
# plot(2*exp(2*t) + exp(-2*t), (t, -2, 2), line_color='g', title='2*exp(2*t) + exp(-2*t)')
# plot_parametric(cos(t), sin(t), (t, 0, 2*pi), line_color='y', title='parametric plot of a circle')


# scipy求微分方程的数值解
def df(y, x):
    return 1 / (1 + x**2) - 2 * y**2  # y' = 1/(1 + x^2) - 2y^2


x = np.linspace(0, 10, 100)
sol = odeint(df, 0, x)
plt.plot(x, sol)
plt.show()


def dy_dt(y, t):
    return np.sin(t**2)


y0 = [1]
t = np.arange(-10, 10, 0.01)
sol = odeint(dy_dt, y0, t)
plt.plot(t, sol)
plt.show()


# 求解高阶微分方程
# odeint是通过把二阶微分转化为一个方程组的形式求解高阶方程的
# y''=20(1-y^2)y'-y
def fvdp(y, t):
    """
    要把y看出一个向量，y = [dy0,dy1,dy2,...]分别表示y的n阶导，那么
    y[0]就是需要求解的函数，y[1]表示一阶导，y[2]表示二阶导，以此类推
    """
    dy1 = y[1]  # y[1]=dy/dt，一阶导                     y[0]表示原函数
    dy2 = 20 * (1 - y[0] ** 2) * y[1] - y[0]  # y[1]表示一阶微分
    # y[0]是最初始，也就是需要求解的函数
    # 注意返回的顺序是[一阶导， 二阶导]，这就形成了一阶微分方程组
    return [dy1, dy2]


# 求解的是一个二阶微分方程，所以输入的时候同时输入原函数y和微分y'
# y[0]表示原函数， y[1]表示一阶微分
# dy1表示一阶微分， dy2表示的是二阶微分
# 可以发现，dy1和y[1]表示的是同一个东西
# 把y''分离变量分离出来： dy2=20*(1-y[0]**2)*y[1]-y[0]
def solve_second_order_ode():
    """
    求解二阶ODE
    """
    x = np.arange(0, 0.25, 0.01)  # 给x规定范围
    y0 = [0.0, 2.0]  # 初值条件
    # 初值[3.0, -5.0]表示y(0)=3,y'(0)=-5
    # 返回y，其中y[:,0]是y[0]的值，就是最终解，y[:,1]是y'(x)的值
    y = odeint(fvdp, y0, x)

    (y1,) = plt.plot(x, y[:, 0], label="y")
    (y1_1,) = plt.plot(x, y[:, 1], label="y‘")
    plt.legend(handles=[y1, y1_1])  # 创建图例

    plt.show()


# solve_second_order_ode()


def f(t, y):
    """
    y''' + y'' - y' + y = cos(t)
    y_0 = 0, y'_0 = pi, y''_0 = 0
    :param t:
    :param y:
    :return:
    """
    dy1 = y[1]
    dy2 = y[2]
    dy3 = -y[0] + dy1 - dy2 - np.cos(t)
    return [dy1, dy2, dy3]


## solve_ivp与odeint的对比使用，一般来说使用solve_ivp更加方便
def solve_high_order_ode():
    """
    求解高阶ODE
    Solve_ivp函数的用法与odeint非常类似，只不过比odeint多了两个参数。一个是t_span参数，表示自变量的取值范围；另一个是method参数，
    可以选择多种不同的数值求解算法。常见的内置方法包括RK45, RK23, DOP853, Radau, BDF等多种方法，通常使用RK45多一些。
    它的使用方法与odeint对比起来很类似。
    但是solve_ivp可以不考虑参数顺序，odeint必须要考虑参数顺序（经验之谈）
    """
    t = np.linspace(0, 6, 1000)  # Time points
    tspan = [0.0, 6.0]  # Time span
    y0 = [0.0, np.pi, 0.0]  # Initial conditions [y(0), y'(0), y''(0)]

    # 注意这里面的odeint和solve_ivp的参数顺序是不一样的，odeint是先传入函数，再传入初值，最后传入时间点
    # 而solve_ivp是先传入函数，再传入时间点，最后传入初值
    # Solve using odeint
    y = odeint(lambda y, t: f(t, y), y0, t)

    # Solve using solve_ivp
    y_ = solve_ivp(f, t_span=tspan, y0=y0, t_eval=t)

    # Plot results
    plt.subplot(211)
    (l1,) = plt.plot(t, y[:, 0], label="y(0) Initial Function")
    (l2,) = plt.plot(t, y[:, 1], label="y(1) The first order of Initial Function")
    (l3,) = plt.plot(t, y[:, 2], label="y(2) The second order of Initial Function")
    plt.legend(handles=[l1, l2, l3])
    plt.grid(True)

    plt.subplot(212)
    (l4,) = plt.plot(y_.t, y_.y[0, :], "r--", label="y(0) Initial Function")
    (l5,) = plt.plot(
        y_.t, y_.y[1, :], "g--", label="y(1) The first order of Initial Function"
    )
    (l6,) = plt.plot(
        y_.t, y_.y[2, :], "b-", label="y(2) The second order of Initial Function"
    )
    plt.legend(handles=[l4, l5, l6])  # 显示图例
    plt.grid(True)
    plt.show()


# Call the function
# solve_high_order_ode()

'''
x''(t) + y'(t) + 3x(t) = cos(2t)
y''(t) -4x'(t) + 3y(t) = sin(2t)
x(0) = 0, x'(0) = 1/5, y(0) = 0, y'(0) = 6/5
'''
def fun(t, w):
    x, y, dx, dy = w
    return [dx, dy, np.cos(2 * t) - 3 * x - dy, np.sin(2 * t) - 3 * y + 4 * dx]

def solve():
    y0 = [0, 0, 1/5, 6/5]
    yy = solve_ivp(fun, (0, 100), y0, t_eval=np.linspace(0, 100, 1000))
    t = yy.t
    print(yy.y.shape)
    x, y = yy.y[0, :], yy.y[1, :]
    dx, dy = yy.y[2, :], yy.y[3, :]
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.plot(t, x, label='x(t)')
    plt.plot(t, y, label='y(t)')
    plt.legend()
    plt.subplot(122)
    plt.plot(t, dx, label="x'(t)")
    plt.plot(t, dy, label="y'(t)")
    plt.legend()
    plt.show()

# solve()

# 偏微分方程求解
# sympy求解偏微分方程
from sympy.abc import x, y
f = Function('f')
eq = -2*f(x, y).diff(x) + 4*f(x, y).diff(y) + 5*f(x, y) - exp(x + 3*y)
ans = pdsolve(eq)
print(ans)