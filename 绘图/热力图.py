import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from MathModels.Plot.styles import mp_seaborn_light

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 设置随机种子以确保结果可重复
np.random.seed(42)

# 生成随机数据
data = np.random.rand(10, 10)  # 生成一个 10x10 的随机矩阵
rows = [f'Row {i+1}' for i in range(10)]  # 行标签
cols = [f'Col {i+1}' for i in range(10)]  # 列标签

# 将数据转换为 DataFrame
df = pd.DataFrame(data, index=rows, columns=cols)

# 创建热力图
plt.figure(figsize=(10, 8))  # 设置图形大小
sns.heatmap(
    df,  # 数据
    annot=True,  # 在热力图上显示数值
    fmt=".2f",  # 数值格式化为两位小数
    cmap='Reds',  # 使用 Reds 颜色映射
    linewidths=0,  # 设置单元格之间的线宽
    # linecolor='black',  # 设置单元格之间的线颜色
    cbar=True,  # 显示颜色条
    cbar_kws={'shrink': 0.8},  # 调整颜色条大小
)

# 设置标题和标签
plt.title('随机生成的热力图示例', fontsize=16, pad=20)
plt.xlabel('列', fontsize=12)
plt.ylabel('行', fontsize=12)

# 显示图形
plt.tight_layout()  # 自动调整布局
plt.show()