from MathModels.Plot.styles import mp_seaborn_light
from matplotlib import pyplot as plt
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

# # 设置x轴刻度
# _xtick_labels = ["{}岁".format(i) for i in x]
# plt.xticks(x, _xtick_labels)
# # plt.yticks(range(0,9))


# 绘制网格,alpha设置网格透明度
plt.grid(alpha=0.5, linestyle=':')

# 添加图例(在指定位置显示线条对应的含义)
plt.legend(loc="upper left")
plt.show()

# seaborn折线图,数据集需要补充
employment = None
plt.gcf().text(.2, .84, 'GENDER', fontsize=40, color='Black') #添加标题
plt.figure(figsize=(14, 7))
plt.style.use('ggplot')
plt.gcf().text(.2, .84, 'GENDER', fontsize=40, color='Black')
sns.set(rc={'xtick.labelsize':17,'ytick.labelsize':10,'axes.labelsize':15, 'axes.grid':False})
sns.lineplot(x='Period', y='Unemployed', hue='Gender', style='Gender',data=employment,
             dashes=False, palette='CMRmap', err_style='bars', ci=70, markers=['o', '>'])
plt.show()