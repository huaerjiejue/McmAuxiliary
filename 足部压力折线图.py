import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import numpy as np
from MathModels.Plot.styles import mp_seaborn_light

# ==================== 数据准备 ====================
data = {
    "Level walking": [48.9, 29.4, 44.7, 43.4, 43.9, 32.8, 52.5],
    "Stair ascent": [45.3, 28.4, 50.6, 50.3, 48.5, 41.2, 34.6],
    "Stair descent": [43.8, 28.5, 57, 53.9, 47.4, 29.7, 21.7]
}
regions = ["Hallux", "2nd-5th toes", "MedFF", "CenFF", "LatFF", "Midfoot", "Heel"]
df = pd.DataFrame(data, index=regions)

# ==================== 样式配置 ====================
# plt.style.use('seaborn-white')
plt.style.use(mp_seaborn_light())
COLORS = ['#1f77b4', '#2ca02c', '#d62728']  # 标准科学配色
MARKERS = ['o', 's', 'D']                   # 圆形、方形、菱形
LINE_STYLES = ['-', '--', '-.']             # 实线、虚线、点划线
FONT = {'family': 'DejaVu Sans', 'size': 10}
plt.rc('font', **FONT)

# ==================== 背景设置 ====================
def set_background(ax, img_path):
    """设置自适应背景"""
    img = Image.open(img_path)
    ax.imshow(img, extent=[-1, len(regions)+1,
                          df.values.min()-5, df.values.max()+5],
             aspect='auto', alpha=0.4, zorder=0)
    ax.set_facecolor((1, 1, 1, 0.7))  # 半透明白色底纹

# ==================== 绘制图表 ====================
fig, ax = plt.subplots(figsize=(12, 7), dpi=150)

# 设置背景（替换为实际图片路径）
# set_background(ax, "contest_code/data/stir_up.jpg")

# 绘制三条折线
for i, (activity, values) in enumerate(data.items()):
    ax.plot(regions, values,
            color=COLORS[i],
            marker=MARKERS[i],
            linestyle=LINE_STYLES[i],
            linewidth=2.5,
            markersize=8,
            markeredgecolor='white',
            markeredgewidth=1.2,
            label=activity,
            zorder=3)

# ==================== 图表优化 ====================
# 坐标轴装饰
ax.set_ylabel('Plantar Pressure (kPa)', fontsize=12, labelpad=15)
ax.set_ylim(15, 65)
ax.yaxis.grid(True, linestyle='--', alpha=0.6)

# X轴标签旋转
# plt.xticks(rotation=30, ha='right', fontsize=10)
plt.xticks(ha = 'center', fontsize=10)

# 添加数据标签
for activity in data:
    for idx, (region, val) in enumerate(zip(regions, data[activity])):
        ax.text(idx, val+1.5, f'{val:.1f}',
                ha='center', va='bottom',
                fontsize=8, color=COLORS[list(data.keys()).index(activity)])

# 专业图例设置
legend = ax.legend(
    title="Gait Mode",
    frameon=True,
    title_fontsize=12,
    fontsize=10,
    loc='upper left',
    bbox_to_anchor=(0.01, 0.98),
    ncol=1,
    shadow=True,
    facecolor='white',
    edgecolor='#CCCCCC'
)
legend.get_frame().set_alpha(0.9)

# ==================== 输出设置 ====================
plt.tight_layout()
plt.savefig('pressure_comparison.png', bbox_inches='tight', transparent=True)
plt.show()