import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 读取数据（假设数据存储在 'data.csv' 文件中）
# 如果你的数据在 Excel 文件中，可以使用 pd.read_excel('data.xlsx')
# data = pd.read_csv('C:\\Users\\emtri\\Desktop\\2x2.xlsx')
data = pd.read_excel('C:\\Users\\emtri\\Desktop\\2x2.xlsx')

# 查看数据
# print(data.head())

# 创建2x2的子图
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# 设置子图标题
fig.suptitle('Dry vs Wet Abrasion Comparison', fontsize=16)

# 设置柱状图的宽度和位置
bar_width = 0.35  # 柱状图宽度
index = np.arange(len(data['Sample']))  # 样本的索引

# 使用 Matplotlib 的 'Paired' 颜色映射
colors = plt.cm.Paired(np.linspace(0, 1, 12))  # 生成12种成对颜色

# 绘制干湿质量损失的柱状图
# axs[0, 0].bar(index - bar_width/2, data.iloc[:, 3], width=bar_width, label='Dry', alpha=0.8, color=colors[0])
# axs[0, 0].bar(index + bar_width/2, data.iloc[:, 7], width=bar_width, label='Wet', alpha=0.8, color=colors[1])
sns.jointplot(x=data.iloc[:,3], y=data.iloc[:,7], kind='scatter')
axs[0, 0].set_title('Mass Loss after 16 Cycles', fontsize=12)
axs[0, 0].set_ylabel('Mass Loss (g/50cm²)', fontsize=10)
axs[0, 0].set_xticks(index)
# axs[0, 0].set_xticklabels(data['Sample'], fontsize=10)
# axs[0, 0].legend()

# 绘制干湿高度损失的柱状图
# axs[0, 1].bar(index - bar_width/2, data.iloc[:, 4], width=bar_width, label='Dry', alpha=0.8, color=colors[2])
# axs[0, 1].bar(index + bar_width/2, data.iloc[:, 8], width=bar_width, label='Wet', alpha=0.8, color=colors[3])
sns.jointplot(x=data.iloc[:,4], y=data.iloc[:,8], kind='scatter')
axs[0, 1].set_title('Height Loss after 16 Cycles', fontsize=12)
axs[0, 1].set_ylabel('Height Loss (cm)', fontsize=10)
axs[0, 1].set_xticks(index)
# axs[0, 1].set_xticklabels(data['Sample'], fontsize=10)
# axs[0, 1].legend()

# 绘制干湿体积损失（基于质量）的柱状图
# axs[1, 0].bar(index - bar_width/2, data.iloc[:, 5], width=bar_width, label='Dry', alpha=0.8, color=colors[4])
# axs[1, 0].bar(index + bar_width/2, data.iloc[:, 9], width=bar_width, label='Wet', alpha=0.8, color=colors[5])
sns.jointplot(x=data.iloc[:,5], y=data.iloc[:,9], kind='scatter')
axs[1, 0].set_title('Volume Loss from Mass', fontsize=12)
axs[1, 0].set_ylabel('Volume Loss (cm³/50cm²)', fontsize=10)
axs[1, 0].set_xticks(index)
# axs[1, 0].set_xticklabels(data['Sample'], fontsize=10)
# axs[1, 0].legend()

# 绘制干湿体积损失（基于高度）的柱状图
# axs[1, 1].bar(index - bar_width/2, data.iloc[:, 6], width=bar_width, label='Dry', alpha=0.8, color=colors[6])
# axs[1, 1].bar(index + bar_width/2, data.iloc[:, 10], width=bar_width, label='Wet', alpha=0.8, color=colors[7])
sns.jointplot(x=data.iloc[:,6], y=data.iloc[:,10], kind='scatter')
axs[1, 1].set_title('Volume Loss from Height', fontsize=12)
axs[1, 1].set_ylabel('Volume Loss (cm³/50cm²)', fontsize=10)
axs[1, 1].set_xticks(index)
# axs[1, 1].set_xticklabels(data['Sample'], fontsize=10)
# axs[1, 1].legend()

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()