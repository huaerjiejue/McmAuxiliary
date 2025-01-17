from MathModels.Plot.styles import mp_seaborn_light
from matplotlib import pyplot as plt

plt.style.use(mp_seaborn_light())
a = ['A', 'B', 'C', 'D', 'E', 'F']
b = [20, 30, 40, 50, 60, 70]

plt.figure(figsize=(20, 8), dpi=80)

# 绘制条形图
plt.bar(range(len(a)), b, width=0.3)

# 设置字符串到x轴
plt.xticks(range(len(a)), a, rotation=90)
plt.rcParams['font.sans-serif'] = ['SimHei', ]

plt.xlabel('类别')
plt.ylabel('数量')
plt.title('柱状图')

plt.show()

# 多条柱状图
a = ["猩球崛起3：终极之战", "敦刻尔克", "蜘蛛侠：英雄归来", "战狼2"]
b_16 = [15746, 312, 4497, 319]
b_15 = [12357, 156, 2045, 168]
b_14 = [2358, 399, 2358, 362]

bar_width = 0.2

x_14 = list(range(len(a)))
x_15 = [i + bar_width for i in x_14]
x_16 = [i + bar_width * 2 for i in x_14]

plt.figure(figsize=(8, 8), dpi=80)

plt.bar(range(len(a)), b_14, width=bar_width, label="9月14日")
plt.bar(x_15, b_15, width=bar_width, label="9月15日")
plt.bar(x_16, b_16, width=bar_width, label="9月16日")

plt.legend()

plt.xticks(x_15, a)

plt.rcParams['font.sans-serif'] = ['SimHei', ]
plt.show()
