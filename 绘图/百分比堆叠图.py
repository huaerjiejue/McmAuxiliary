import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter
import seaborn as sns

def load_data():
    # 示例数据（用于演示）
    return pd.DataFrame({
        'Category': ['A', 'B', 'C', 'D', 'E'],
        'Fair': [10, 20, 30, 40, 50],
        'Good': [20, 30, 40, 50, 60],
        'Very Good': [30, 40, 50, 60, 70],
        'Premium': [40, 50, 60, 70, 80],
        'Ideal': [50, 60, 70, 80, 90]
    })

def plot_stacked_bar_chart(df):
    # colors = ['#ADFEDC', '#4EFEB3', '#02F78E', '#02CB74', '#019858']
    colors = sns.color_palette("Greens", 5)
    # labels = df.columns[1:].tolist()  # 获取列标签
    labels = pd.unique(df['Category'])  # 获取类别标签

    y_values = df.iloc[0:5, 1:].values  # 获取所有行的数据
    data = y_values.T  # 转置数组，使得每个类别的数据成为一个列

    x = range(len(labels))  # x轴标签
    bottom_y = np.zeros(len(labels))  # 初始化底部y值为0

    figure, ax = plt.subplots()
    for i, color in enumerate(colors):
        y = data[i] / data[i].sum()  # 计算百分比
        ax.bar(x, y, width=0.5, color=color, bottom=bottom_y, edgecolor='gray', label=labels[i])
        bottom_y += y  # 更新底部y值以进行堆积

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    legend_labels = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, legend_labels)]
    ax.legend(handles=patches, ncol=1, loc='upper right')

    ax.yaxis.set_major_formatter(PercentFormatter(1))

    for i in range(1, 11):
        ax.axhline(y=i / 10, linestyle='dashed', color='black', linewidth=0.5)

    plt.rcParams['font.sans-serif'] = ['SimHei',]
    ax.set_title('百分比堆积柱状图', fontsize=13)
    ax.set_ylabel('百分比', fontsize=13)
    ax.set_xlabel('类别', fontsize=13)

    plt.show()

if __name__ == "__main__":
    df = load_data()
    plot_stacked_bar_chart(df)