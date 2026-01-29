import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 石阶尺寸参数
W, L = 2.0, 1.5
x = np.linspace(0, W, 100)
y = np.linspace(0, L, 100)
X, Y = np.meshgrid(x, y)

# 凹陷中心点（宽度1/2，长度1/3）
center_x, center_y = W/2, 2*L/3

# 生成凹陷分布（反向高斯函数）
sigma = 0.5
distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
gaussian = 1 - np.exp(-distance**2 / (2 * sigma**2))  # 中心值最低（凹陷最深）

# 创建3D图形
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# 使用inferno色阶（中心深色，边缘亮黄色）
surf = ax.plot_surface(
    X, Y, gaussian,
    cmap='inferno',          # 核心修改：inferno色阶
    facecolors=plt.cm.inferno(gaussian),
    rstride=2, cstride=2,
    alpha=0.9,               # 降低透明度以突出颜色对比
    linewidth=0,
    antialiased=True
)

# 标记凹陷中心（白色以在inferno色阶中突出）
ax.scatter(
    [center_x], [center_y], [gaussian.min()],
    c='white', s=100, label='Depression Center', edgecolors='black', depthshade=False
)

# 设置视角和光照
ax.view_init(elev=40, azim=-60)
ax.set_zlim(gaussian.min()-0.1, gaussian.max())

# 标签和颜色条
ax.set_xlabel('Width (m)', fontsize=12)
ax.set_ylabel('Length (m)', fontsize=12)
ax.set_zlabel('Depression Depth', fontsize=12)
ax.set_title('3D Depression Distribution (Inferno Colormap)', fontsize=14)
fig.colorbar(surf, shrink=0.5, label='Depth Intensity')

# 移除背景网格和面板
ax.grid(False)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
plt.legend()
plt.show()