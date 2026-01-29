import matplotlib.pyplot as plt
import numpy as np

# 数据准备
regions = ['Hallux', '2nd-5th toes', 'MedFF', 'CentFF', 'LatFF', 'Midfoot', 'Heel']
level = [48.9, 29.4, 44.7, 43.4, 43.9, 32.8, 52.5]
ascent = [45.3, 28.4, 50.6, 50.3, 48.5, 41.2, 34.6]
descent = [43.8, 28.5, 57.0, 53.9, 47.4, 29.7, 21.7]

# 标准差数据
std_level = [13.9, 9.0, 12.8, 9.2, 9.3, 8.5, 9.6]
std_ascent = [17.0, 11.8, 17.2, 15.9, 14.9, 12.1, 15.2]
std_descent = [18.0, 8.7, 16.1, 12.2, 12.8, 12.3, 10.0]

# 科学绘图风格设置
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300
})


def create_comparison_plot(compare_mode, title_suffix):
    fig, ax = plt.subplots(figsize=(8, 5))

    # 选择对比模式
    if compare_mode == "ascent":
        compare_data = ascent
        std_compare = std_ascent
        color = '#2ca02c'  # 绿色系
    else:
        compare_data = descent
        std_compare = std_descent
        color = '#d62728'  # 红色系

    # 绘制曲线
    x = np.arange(len(regions))
    ax.plot(x, level, 'o-', color='#1f77b4', label='Level Walking', linewidth=2, markersize=8)
    ax.plot(x, compare_data, 's--', color=color, label=title_suffix, linewidth=2, markersize=8)

    # 添加误差条
    ax.errorbar(x, level, yerr=std_level, fmt='none', ecolor='#1f77b4', elinewidth=1, capsize=4)
    ax.errorbar(x, compare_data, yerr=std_compare, fmt='none', ecolor=color, elinewidth=1, capsize=4)

    # 坐标轴设置
    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=45, ha='right')
    ax.set_ylabel('Pressure (kPa)')
    ax.set_title(f'Plantar Pressure: Level Walking vs {title_suffix}', pad=20)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(frameon=True, loc='upper right')

    # 科学图表调整
    plt.tight_layout()
    plt.savefig(f'pressure_comparison_{compare_mode}.png', bbox_inches='tight')
    plt.close()


# 生成两张图片
create_comparison_plot("ascent", "Stair Ascent")
create_comparison_plot("descent", "Stair Descent")