import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from heapq import heappush, heappop

# ========================== 参数配置 ==========================
STAIR_STEPS = 20  # 台阶数量
STAIR_WIDTH = 20  # 台阶宽度（单元格数）
CELL_SIZE = 0.53  # 单元格边长 (m)
SIM_STEPS = 200  # 模拟步数
NUM_PEDESTRIANS = 200  # 初始行人数量

# 材料参数
MATERIAL_K = 1e-8  # 磨损系数 (m³/N·m)
MATERIAL_H = 70  # 材料硬度 (Pa)
BODY_WEIGHT = 60  # 平均体重 (kg)

# 行人参数
MEAN_VELOCITY = 1.0  # 平均自由流速 (m/s)
VELOCITY_STD = 0.3  # 速度标准差
UP_RATIO = 0.4  # 上行比例

# 颜色配置
CMAP_DENSITY = 'YlOrRd'  # 改为单方向渐变色
CMAP_WEAR = 'plasma'     # 高对比度配色
WEAR_UNIT = 'mm'         # 明确磨损单位

# ====================== 六边形网格系统 ========================
class HexGrid:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols), dtype=int)  # 0=空, 1=行人, 2=障碍物, 3=目标
        self.wear = np.zeros((rows, cols))  # 磨损深度记录

        # 六边形邻居方向 (行, 列偏移)
        self.directions = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]

        # 初始化障碍物（楼梯两侧）
        self.grid[:, 0] = 2
        self.grid[:, -1] = 2

        # 设置目标位置（顶端和底端中心）
        self.target_up = (0, cols // 2)
        self.target_down = (rows - 1, cols // 2)
        self.grid[self.target_up] = 3
        self.grid[self.target_down] = 3

        # 生成势场
        self.floor_field_up = self.generate_floor_field(self.target_up)
        self.floor_field_down = self.generate_floor_field(self.target_down)

    def generate_floor_field(self, target):
        """使用Dijkstra算法生成最短路径势场"""
        q = []
        field = np.full((self.rows, self.cols), np.inf)
        field[target] = 0
        heappush(q, (0, target))

        while q:
            dist, (r, c) = heappop(q)
            for dr, dc in self.directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if self.grid[nr, nc] == 2:  # 跳过障碍物
                        continue
                    new_dist = dist + 1
                    if new_dist < field[nr, nc]:
                        field[nr, nc] = new_dist
                        heappush(q, (new_dist, (nr, nc)))
        return field

    def get_neighbors(self, pos):
        """获取有效邻居位置"""
        r, c = pos
        neighbors = []
        for dr, dc in self.directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if self.grid[nr, nc] != 2:  # 排除障碍物
                    neighbors.append((nr, nc))
        return neighbors


# ======================== 行人对象 ===========================
class Pedestrian:
    def __init__(self, grid, up_ratio):
        # 初始位置
        if np.random.rand() < up_ratio:  # 上行
            self.pos = (grid.rows - 1, np.random.randint(1, grid.cols - 1))
            self.target = grid.target_up
            self.direction = 'up'
        else:  # 下行
            self.pos = (0, np.random.randint(1, grid.cols - 1))
            self.target = grid.target_down
            self.direction = 'down'

        # 动态属性
        self.v0 = np.random.normal(MEAN_VELOCITY, VELOCITY_STD)
        self.active = True


# ======================== 模拟引擎 ============================
def simulate(up_redio):
    # 初始化
    grid = HexGrid(STAIR_STEPS, STAIR_WIDTH)
    pedestrians = [Pedestrian(grid, up_redio) for _ in range(NUM_PEDESTRIANS)]
    density = np.zeros((grid.rows, grid.cols))  # 密度累积数组

    for step in range(SIM_STEPS):
        # === 行人移动阶段 ===
        move_plans = {}

        # 阶段1：收集所有移动计划
        for p in pedestrians:
            if not p.active: continue

            # 获取当前势场
            ff = grid.floor_field_up if p.direction == 'up' else grid.floor_field_down
            current_pot = ff[p.pos]

            # 选择最佳移动
            best_move = None
            min_pot = np.inf
            for n in grid.get_neighbors(p.pos):
                pot = ff[n]
                if pot < min_pot and pot <= current_pot:
                    best_move = n
                    min_pot = pot

            # 冲突检测
            if best_move:
                if best_move in move_plans:
                    # 优先权：同向 > 速度快
                    if p.v0 > move_plans[best_move].v0:
                        move_plans[best_move].active = False  # 取消原有移动
                        move_plans[best_move] = p
                else:
                    move_plans[best_move] = p

        # 阶段2：执行移动
        wear_updates = {}
        for new_pos, p in move_plans.items():
            old_pos = p.pos

            # 更新网格状态
            grid.grid[old_pos] = 0
            grid.grid[new_pos] = 1
            p.pos = new_pos

            # 计算磨损
            step_length = CELL_SIZE * np.cos(np.radians(30))
            Fn = BODY_WEIGHT * 9.8
            delta_wear = (MATERIAL_K * Fn * step_length / MATERIAL_H) * 1000  # mm
            # hex_area = (3 * np.sqrt(3) / 2) * (CELL_SIZE ** 2)  # 六边形面积公式
            # delta_wear_volume = MATERIAL_K * Fn * step_length / MATERIAL_H
            # delta_wear_depth = (delta_wear_volume / hex_area) * 1000  # 转换为毫米

            if old_pos in wear_updates:
                wear_updates[old_pos] += delta_wear
                # wear_updates[old_pos] += delta_wear_depth
            else:
                wear_updates[old_pos] = delta_wear
                # wear_updates[old_pos] = delta_wear_depth

            # 到达检测
            if new_pos == p.target:
                p.active = False
                grid.grid[new_pos] = 3  # 保持目标点状态

        # 更新磨损数据
        for pos, wear in wear_updates.items():
            grid.wear[pos] += wear

        # === 实时密度统计 ===
        for p in pedestrians:
            if p.active:
                r, c = p.pos
                density[r, c] += 1

        # 动态更新势场（每10步更新一次）
        if step % 10 == 0:
            grid.floor_field_up = grid.generate_floor_field(grid.target_up)
            grid.floor_field_down = grid.generate_floor_field(grid.target_down)

    return density / SIM_STEPS, grid.wear

    # ====================== 结果可视化 ======================
    # plt.figure(figsize=(14, 6), dpi=100)
    #
    # # --- 子图1：行人密度分布 ---
    # plt.subplot(1, 2, 1)
    #
    # # 计算绝对密度（人数/单元格）
    #
    # for p in pedestrians:
    #     if p.active:
    #         r, c = p.pos
    #         density[r, c] += 1
    #
    # # 可视化设置
    # im1 = plt.imshow(
    #     density.T,
    #     cmap=CMAP_DENSITY,
    #     origin='lower',
    #     extent=[0, STAIR_STEPS * CELL_SIZE, 0, STAIR_WIDTH * CELL_SIZE],
    #     interpolation='nearest'
    # )
    #
    # # 标注物理坐标
    # plt.xticks(
    #     np.linspace(0, STAIR_STEPS * CELL_SIZE, 5),
    #     labels=[f"{x:.1f}" for x in np.linspace(0, STAIR_STEPS * CELL_SIZE, 5)]
    # )
    # plt.yticks(
    #     np.linspace(0, STAIR_WIDTH * CELL_SIZE, 5),
    #     labels=[f"{y:.1f}" for y in np.linspace(0, STAIR_WIDTH * CELL_SIZE, 5)]
    # )
    #
    # plt.colorbar(im1, label='Pedestrian Density (people/unit)', shrink=0.8)
    # plt.xlabel("Stair Longitudinal Distance (m)")
    # plt.ylabel("Stair Lateral Distance (m)")
    # plt.title("Pedestrian Density Distribution (Non-negative Values)")
    #
    # # --- 子图2：磨损深度分布 ---
    # plt.subplot(1, 2, 2)
    #
    # # 数据清洗（过滤负值）
    # wear_data = np.where(grid.wear < 0, 0, grid.wear)
    #
    # # 动态调整颜色范围（排除前1%极端值）
    # vmax = np.percentile(wear_data[wear_data > 0], 99) if np.any(wear_data > 0) else 0.1
    #
    # im2 = plt.imshow(
    #     wear_data.T,
    #     cmap=CMAP_WEAR,
    #     origin='lower',
    #     vmin=0,
    #     vmax=vmax,
    #     extent=[0, STAIR_STEPS * CELL_SIZE, 0, STAIR_WIDTH * CELL_SIZE],
    #     interpolation='nearest'
    # )
    #
    # # 标注物理坐标
    # plt.xticks(
    #     np.linspace(0, STAIR_STEPS * CELL_SIZE, 5),
    #     labels=[f"{x:.1f}" for x in np.linspace(0, STAIR_STEPS * CELL_SIZE, 5)]
    # )
    # plt.yticks(
    #     np.linspace(0, STAIR_WIDTH * CELL_SIZE, 5),
    #     labels=[f"{y:.1f}" for y in np.linspace(0, STAIR_WIDTH * CELL_SIZE, 5)]
    # )
    #
    # cbar = plt.colorbar(im2, label=f'Cumulative Wear Depth ({WEAR_UNIT})', shrink=0.8)
    # cbar.formatter.set_powerlimits((0, 0))  # Disable scientific notation
    # plt.xlabel("Stair Longitudinal Distance (m)")
    # plt.ylabel("Stair Lateral Distance (m)")
    # plt.title(f"Material Wear Distribution (Dynamic Range: 0~{vmax:.2f} {WEAR_UNIT})")
    #
    # # --- Global Settings ---
    # plt.tight_layout()
    # plt.suptitle(
    #     f"Stair Pedestrian Flow and Wear Simulation Results (Steps={SIM_STEPS}, Pedestrians={NUM_PEDESTRIANS})",
    #     y=1.02,
    #     fontsize=14
    # )
    # plt.show()


def visualize_results():
    ratios = [0.3, 0.5, 0.7]
    fig, axs = plt.subplots(2, 3, figsize=(18, 12), dpi=100)  # 改为2行3列

    # 预计算所有数据
    results = []
    for ratio in ratios:
        density, wear = simulate(ratio)  # 假设simulate函数返回密度和磨损数据
        results.append((density, wear))

    # print(results)

    # 统一颜色范围
    density_max = np.max([np.max(d) for d, _ in results])
    wear_max = np.max([np.percentile(w[w > 0], 99) for _, w in results if np.any(w > 0)])

    # ===== 第一行：行人密度 =====
    for col, (ratio, (density, _)) in enumerate(zip(ratios, results)):
        ax = axs[0, col]
        im = ax.imshow(density.T,
                       cmap=CMAP_DENSITY,
                       vmin=0,
                       vmax=density_max,
                       extent=[0, STAIR_STEPS * CELL_SIZE, 0, STAIR_WIDTH * CELL_SIZE])
        ax.set_title(f"Density (UP={ratio})", fontsize=12)
        ax.set_xlabel("Longitudinal (m)")
        if col == 0:
            ax.set_ylabel("Lateral (m)")
        fig.colorbar(im, ax=ax, label='People/cell', shrink=0.8)

    # ===== 第二行：磨损分布 =====
    for col, (ratio, (_, wear)) in enumerate(zip(ratios, results)):
        ax = axs[1, col]
        wear_data = np.where(wear < 0, 0, wear)
        im = ax.imshow(wear_data.T,
                       cmap=CMAP_WEAR,
                       vmin=0,
                       vmax=wear_max,
                       extent=[0, STAIR_STEPS * CELL_SIZE, 0, STAIR_WIDTH * CELL_SIZE])
        ax.set_title(f"Wear (UP={ratio})", fontsize=12)
        ax.set_xlabel("Longitudinal (m)")
        if col == 0:
            ax.set_ylabel("Lateral (m)")
        cbar = fig.colorbar(im, ax=ax, label=f'Wear ({WEAR_UNIT})', shrink=0.8)
        cbar.formatter.set_powerlimits((0, 0))

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    # plt.suptitle("Pedestrian Flow and Wear Analysis",
    #              fontsize=16, y=0.98)
    plt.show()


# def check_results():
#
#     # ========== 遍历不同人数（固定速度） ==========
#     velocities = []  # 存储速度参数（此处固定）
#     pedestrian_counts = np.arange(1000, 10001, 100)  # 100到1000，间隔100
#     density_means = []
#     wear_means = []
#
#     MEAN_VELOCITY = 1.0  # 固定速度参数
#
#     for num in pedestrian_counts:
#         # global NUM_PEDESTRIANS
#         NUM_PEDESTRIANS = num  # 修改全局参数
#         density, wear = simulate(0.5)
#         density_means.append(density.mean())
#         wear_means.append(wear.max()*10000)  # 转换为毫米
#
#     # 绘制人数变化影响图
#     # plt.figure(figsize=(10, 4))
#     # plt.subplot(1, 2, 1)
#     # plt.plot(pedestrian_counts, density_means, 'bo-', label='Density')
#     # plt.plot(pedestrian_counts, wear_means, 'rs-', label='Wear')
#     # plt.xlabel('Number of Pedestrians')
#     # plt.ylabel('Mean Value')
#     # plt.title(f'Fixed Velocity={MEAN_VELOCITY}')
#     # plt.legend()
#     # plt.grid(True)
#     print(np.array(density_means).mean())
#     print(np.array(wear_means).mean())
#
#     # ========== 遍历不同速度（固定人数） ==========
#     NUM_PEDESTRIANS = 100  # 固定人数参数
#     velocity_range = np.linspace(1.0, 10.0, 100)  # 1到10，均匀取10个点
#     density_means_v = []
#     wear_means_v = []
#
#     for v in velocity_range:
#         # global MEAN_VELOCITY
#         MEAN_VELOCITY = v  # 修改全局参数
#         density, wear = simulate(0.5)
#         density_means_v.append(density.mean())
#         wear_means_v.append(wear.max() * 10000)  # 转换为毫米
#
#     # 绘制速度变化影响图
#     # plt.subplot(1, 2, 2)
#     # plt.plot(velocity_range, density_means_v, 'bo-', label='Density')
#     # plt.plot(velocity_range, wear_means_v, 'rs-', label='Wear')
#     # plt.xlabel('Mean Velocity')
#     # plt.ylabel('Mean Value')
#     # plt.title(f'Fixed Pedestrians={NUM_PEDESTRIANS}')
#     # plt.legend()
#     # plt.grid(True)
#     #
#     # plt.tight_layout()
#     # plt.show()
#     print(np.array(density_means_v).mean())
#     print(np.array(wear_means_v).mean())

def check_results():
    min_peds = 100
    max_peds = 10**4
    min_velocity = 1.0
    max_velocity = 10.0
    ans = -1
    best_peds = min_peds
    best_velocity = min_velocity
    for peds in range(min_peds, max_peds+1, 100):
        for velocity in np.linspace(min_velocity, max_velocity, 100):
            MEAN_VELOCITY = velocity
            NUM_PEDESTRIANS = peds
            density, wear = simulate(0.5)
            result = -np.array(wear).max()
            if result > ans:
                ans = result
                best_peds = peds
                best_velocity = velocity
    print(f"最佳人数：{best_peds}，最佳速度：{best_velocity}，最佳结果：{ans}")

# 运行模拟
if __name__ == "__main__":
    # simulate()
    # visualize_results()
    check_results()
    # for UP_RATIO in [0.3, 0.5, 0.8]:
    #     simulate()