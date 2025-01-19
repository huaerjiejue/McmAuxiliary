from MathModels.Plot.styles import mp_seaborn_light
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

plt.style.use(mp_seaborn_light())

x = range(11, 31)
y_1 = [1, 0, 1, 1, 2, 4, 3, 2, 3, 4, 4, 5, 6, 5, 4, 3, 3, 1, 1, 1]
y_2 = [1, 0, 3, 1, 2, 2, 3, 3, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# 设置图形大小
plt.figure(figsize=(20, 8), dpi=80)

# 画两条线，并写明哪条线表示什么,设置线条样式
plt.plot(x, y_1, label="score1", color="coral", linewidth=5)
# 为线条添加阴影
y1_low = [i+1 for i in y_1]
y1_high = [i-1 for i in y_1]
plt.fill_between(x, y1_low, y1_high, color='coral', alpha=0.3)
plt.plot(x, y_2, label="score2", color="cyan", linestyle='--')


# 示例数据
x = np.arange(2006, 2015)  # 年份
y = np.array([100, 150, 200, 250, 300, 350, 400, 450, 500])  # 假设的数值

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='o', label='Data')

# 添加箭头标注
highlight_year = 2010  # 需要标注的年份
highlight_index = np.where(x == highlight_year)[0][0]  # 找到对应的索引
highlight_value = y[highlight_index]  # 找到对应的值

plt.annotate(
    f'Important Point ({highlight_year}, {highlight_value})',  # 标注文本
    xy=(highlight_year, highlight_value),  # 箭头指向的坐标
    xytext=(highlight_year - 2, highlight_value + 50),  # 文本位置
    arrowprops=dict(facecolor="black", edgecolor="black", arrowstyle='->'),  # 箭头样式
    fontsize=12,
    color='black'
)

# 添加标题和标签
plt.title('Data Over Years with Annotation', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Value', fontsize=12)

# 添加图例
plt.legend()

# 显示图表
plt.show()