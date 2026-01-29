import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tqdm import tqdm, trange

# from contest_code.第一问求解 import stone_stirs

upstirs = np.ones([15, 6])
downstirs = np.ones([15, 6])
# stone_stirs = np.zeros([20, 50])

upstirs[0:3, 0:2] = 45.3
upstirs[0:3, 2:6] = 28.4
upstirs[3:7, 0:2] = 50.6
upstirs[3:7, 2:4] = 50.3
upstirs[3:7, 4:6] = 48.5
upstirs[7:11] = 41.2
upstirs[11:15] = 34.6

downstirs[0:4] = 21.7
downstirs[4:8] = 29.7
downstirs[8:12,0:2] = 47.4
downstirs[8:12,2:4] = 53.9
downstirs[8:12,4:6] = 57
downstirs[12:15,0:4] = 28.5
downstirs[12:15,4:6] = 43.8

def trunc_normal_int(min_val=5, max_val=15, mean=10, std_dev=2):
    # 计算截断范围（标准化到标准正态分布）
    a = (min_val - 0.5 - mean) / std_dev
    b = (max_val + 0.5 - mean) / std_dev
    # 生成截断正态分布的随机数
    num = truncnorm.rvs(a, b, loc=mean, scale=std_dev)
    return int(np.round(num))

# for i in range(500):
#     x = trunc_normal_int(5, 15, 10, 2)
#     y = trunc_normal_int(20, 50, 35, 3)
#     if np.random.rand() < 0.8:
#         stone_stirs[x:20, y-3:y+3] += upstirs[0:x, 0:6]
#     else:
#         stone_stirs[0:x, y-3:y+3] += downstirs[20-x:20, 0:6]

def safe_slice_add(target, source, t_row_start, t_row_end, t_col_start, t_col_end):
    """安全切片累加函数"""
    # 调整目标行范围
    t_row_start = max(0, t_row_start)
    t_row_end = min(target.shape[0], t_row_end)

    # 调整目标列范围
    t_col_start = max(0, t_col_start)
    t_col_end = min(target.shape[1], t_col_end)

    # 计算需要的行数和列数
    need_rows = t_row_end - t_row_start
    need_cols = t_col_end - t_col_start

    # 计算源数组的有效切片
    s_row_end = min(source.shape[0], need_rows)
    s_col_end = min(source.shape[1], need_cols)

    # 执行累加操作
    if need_rows > 0 and need_cols > 0:
        target[t_row_start:t_row_end, t_col_start:t_col_end] += source[0:s_row_end, 0:s_col_end]


# 主循环
# for _ in trange(5000):
#     x = trunc_normal_int(5, 15, 10, 3)
#     y = trunc_normal_int(0, 50, 25, 8)
#
#     if np.random.rand() < 0.5:
#         # 上行操作逻辑
#         safe_slice_add(
#             target=stone_stirs,
#             source=upstirs,
#             t_row_start=x,
#             t_row_end=20,  # 原代码中的x:20
#             t_col_start=y - 3,
#             t_col_end=y + 3,
#         )
#     else:
#         # 下行操作逻辑
#         safe_slice_add(
#             target=stone_stirs,
#             source=downstirs,
#             t_row_start=0,
#             t_row_end=x,  # 原代码中的0:x
#             t_col_start=y - 3,
#             t_col_end=y + 3,
#         )

def simulate_stair_wear(
    n_iter=5000,        # 总迭代次数
    up_prob=0.8,        # 上行概率(0-1)
    x_mean=10,          # x的均值
    x_std=3,            # x的标准差
    x_min=5,            # x的最小值
    x_max=15,           # x的最大值
    y_mean=25,          # y的均值
    y_std=8,            # y的标准差
    y_min=0,            # y的最小值
    y_max=50,            # y的最大值
):
    stone_stirs = np.zeros([20, 50])
    for _ in trange(n_iter):
        x = trunc_normal_int(x_min, x_max, x_mean, x_std)
        y = trunc_normal_int(y_min, y_max, y_mean, y_std)

        if np.random.rand() < up_prob:
            # 上行操作逻辑
            safe_slice_add(
                target=stone_stirs,
                source=upstirs,
                t_row_start=x,
                t_row_end=20,  # 原代码中的x:20
                t_col_start=y - 3,
                t_col_end=y + 3,
            )
        else:
            # 下行操作逻辑
            safe_slice_add(
                target=stone_stirs,
                source=downstirs,
                t_row_start=0,
                t_row_end=x,  # 原代码中的0:x
                t_col_start=y - 3,
                t_col_end=y + 3,
            )

    return stone_stirs

def smooth_heatmap(data, sigma=1.2):
    """高斯平滑处理"""
    # 按行标准化
    row_sums = data.sum(axis=1)
    row_sums[row_sums == 0] = 1  # 避免除零错误
    normalized = data / row_sums[:, np.newaxis]

    # 高斯滤波
    smoothed = gaussian_filter(normalized, sigma=sigma)

    # 还原量级
    return smoothed * row_sums[:, np.newaxis]

# fig, axs = plt.subplots(3, 3, figsize=(16, 16))
# stone_stirs = []
# stone_stirs.append(smooth_heatmap(simulate_stair_wear(y_mean=15, up_prob=0.8)))
# stone_stirs.append(smooth_heatmap(simulate_stair_wear(y_mean=25, up_prob=0.8)))
# stone_stirs.append(smooth_heatmap(simulate_stair_wear(y_mean=35, up_prob=0.8)))
# stone_stirs.append(smooth_heatmap(simulate_stair_wear(y_mean=15, up_prob=0.5), sigma=1.5))
# stone_stirs.append(smooth_heatmap(simulate_stair_wear(y_mean=25, up_prob=0.5), sigma=1.5))
# stone_stirs.append(smooth_heatmap(simulate_stair_wear(y_mean=35, up_prob=0.5), sigma=1.5))
# stone_stirs.append(smooth_heatmap(simulate_stair_wear(y_mean=15, up_prob=0.2), sigma=1.8))
# stone_stirs.append(smooth_heatmap(simulate_stair_wear(y_mean=25, up_prob=0.2), sigma=1.8))
# stone_stirs.append(smooth_heatmap(simulate_stair_wear(y_mean=35, up_prob=0.2), sigma=1.8))
# for i in range(3):
#     for j in range(3):
#         axs[i, j].imshow(stone_stirs[i*3+j], cmap='hot', aspect='auto')
#         # axs[i, j].set_title(f'x_mean={7+i*3}, up_prob={0.8-0.3*j}')
# plt.show()


# 同行串行
fig, axs = plt.subplots(3, 3, figsize=(16, 16))
stone_stirs = []
stone_stirs.append(smooth_heatmap(simulate_stair_wear(up_prob=0.8, y_std=8)))
stone_stirs.append(smooth_heatmap(simulate_stair_wear(y_std=10, up_prob=0.8)))
stone_stirs.append(smooth_heatmap(simulate_stair_wear(up_prob=0.8, y_std=13)))
stone_stirs.append(smooth_heatmap(simulate_stair_wear(up_prob=0.5, y_std=8), sigma=1.5))
stone_stirs.append(smooth_heatmap(simulate_stair_wear(up_prob=0.5, y_std=10), sigma=1.5))
stone_stirs.append(smooth_heatmap(simulate_stair_wear(up_prob=0.5, y_std=13), sigma=1.5))
stone_stirs.append(smooth_heatmap(simulate_stair_wear(up_prob=0.2, y_std=8), sigma=1.8))
stone_stirs.append(smooth_heatmap(simulate_stair_wear(up_prob=0.2, y_std=10), sigma=1.8))
stone_stirs.append(smooth_heatmap(simulate_stair_wear(up_prob=0.2, y_std=13), sigma=1.8))

for i in range(3):
    for j in range(3):
        axs[i, j].imshow(stone_stirs[i*3+j], cmap='hot', aspect='auto')
        # axs[i, j].set_title(f'y_std={8+i*2}, up_prob={0.8-0.3*j}')
plt.show()