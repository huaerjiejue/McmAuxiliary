import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import MathModels.Plot.styles as styles

plt.style.use(styles.mp_seaborn_light())

data = sns.load_dataset("tips")

# 创建回归拟合图
plt.figure(figsize=(10, 8))
sns.regplot(
    x='total_bill',  # x 轴数据
    y='tip',  # y 轴数据
    data=data,  # 数据
    scatter_kws={'s': 100, 'alpha': 1},  # 散点图参数
    line_kws={'color': '#FF5722', 'linewidth': 2},  # 回归线参数
    marker='>',  # 散点图标记
    color='blue',  # 散点图颜色
    logx=True,  # 时候开启log拟合
)
plt.title('餐厅账单与小费回归拟合图', fontsize=16, pad=20)
plt.xlabel('账单金额', fontsize=12)
plt.ylabel('小费金额', fontsize=12)
plt.tight_layout()
plt.show()

# 拟合多条线
data = sns.load_dataset("anscombe")
sns.lmplot(
    x='x',  # x 轴数据
    y='y',  # y 轴数据
    data=data,  # 数据
    # col='dataset',  # 按照 dataset 分组  如果选择了col，那么就会分组画图，如果没有选择col，那么就会画在一张图上
    hue='dataset',  # 按照 dataset 分组
    # palette='Set1',  # 颜色
    scatter_kws={'s': 100, 'alpha': 1},  # 散点图参数
    line_kws={'linewidth': 2},  # 回归线参数
)
plt.show()