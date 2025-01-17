import matplotlib.pyplot as plt
from MathModels.Plot.styles import mp_seaborn_light
import seaborn as sns

plt.style.use(mp_seaborn_light())

data = [2052380, 11315444, 20435242, 7456627, 3014264, 1972395, 185028]
# 数据标签
labels = ['none', 'primary', 'junior', 'senior', 'specialties', 'bachelor', 'master']
# 各区域颜色
# colors = ['red', 'orange', 'yellow', 'green', 'purple', 'blue', 'black']

# 设置突出模块偏移值
expodes = (0, 0, 0.05, 0, 0, 0, 0)
# 设置绘图属性并绘图
plt.pie(data, explode=expodes, labels=labels, shadow=True, startangle=90, autopct='%1.1f%%')
## 用于显示为一个长宽相等的饼图
plt.axis('equal')
# 保存并显示
# plt.savefig('picture/step3/fig3.png')
plt.show()

# 环形图
sns.set_theme(font_scale = 1.2)
plt.figure(figsize=(8,8))
plt.pie(
    x=data, labels=labels,
    colors=sns.color_palette('Set2'),
    # startangle=90,
    # show percentages
    autopct='%1.2f%%',
    # move the percentage inside the arcs
    pctdistance=0.80,
    # add space between the arcs
    explode=expodes
)
### Add a hole in the pie
hole = plt.Circle((0, 0), 0.65, facecolor='white')
plt.gcf().gca().add_artist(hole)

plt.show()