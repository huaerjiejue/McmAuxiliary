import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def plot_discrete(file_path, column_name, target_path):
    # 创建目标文件夹
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # 读取数据
    data = pd.read_csv(file_path)

    # 设置科研绘图风格
    plt.style.use('seaborn-v0_8-paper')  # 使用简洁的科研风格
    sns.set_palette("muted")  # 使用低饱和度配色

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)  # 高分辨率

    # 统计每个离散值的频次
    value_counts = data[column_name].value_counts().sort_index()

    # 绘制柱状图 - 使用低饱和度的蓝灰色
    bars = ax.bar(value_counts.index,
                  value_counts.values,
                  color='#5B8FA3',  # 低饱和度蓝灰色
                  edgecolor='#2C3E50',  # 深灰色边框
                  linewidth=1.2,
                  alpha=0.85)

    # 在柱状图上方添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{int(height)}',
                ha='center', va='bottom',
                fontsize=9, color='#2C3E50')

    # 设置标题和标签 - 使用更专业的字体大小和样式
    # ax.set_title(f'Distribution of {column_name}',
    #              fontsize=14, fontweight='bold', pad=20, color='#2C3E50')
    ax.set_xlabel(column_name, fontsize=12, fontweight='semibold', color='#2C3E50')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='semibold', color='#2C3E50')

    # 添加网格线 - 仅显示水平网格，增加可读性
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='#7F8C8D')
    ax.set_axisbelow(True)  # 网格线置于底层

    # 优化刻度
    ax.tick_params(axis='both', labelsize=10, colors='#2C3E50')

    # 添加统计信息文本框
    stats_text = f'N = {len(data)}\nUnique values = {data[column_name].nunique()}'
    ax.text(0.98, 0.97, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white',
                      edgecolor='#BDC3C7', alpha=0.8, linewidth=1))

    # 自动调整布局，避免标签被截断
    plt.tight_layout()

    # 保存图像 - 高质量PNG和矢量PDF两种格式
    base_name = f'{column_name}_distribution'
    plt.savefig(os.path.join(target_path, f'{base_name}_dis.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(target_path, f'{base_name}_dis.pdf'),
                bbox_inches='tight', facecolor='white')

    plt.close()

    print(f"图表已保存至: {target_path}")
    print(f"  - {base_name}.png (高分辨率位图)")
    print(f"  - {base_name}.pdf (矢量图)")


def plot_continuous(file_path, column_name, target_path, bins=30):
    """
    绘制连续型数据的分布图

    Parameters:
    -----------
    file_path : str
        CSV文件路径
    column_name : str
        要绘制的列名
    target_path : str
        保存图片的目标路径
    bins : int, optional
        直方图的分箱数量，默认为30
    """
    # 创建目标文件夹
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # 读取数据
    data = pd.read_csv(file_path)

    # 删除缺失值
    clean_data = data[column_name].dropna()

    # 设置科研绘图风格
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette("muted")

    # 创建图形 - 使用subplot来组合直方图和核密度估计
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # 绘制直方图
    n, bins_edges, patches = ax.hist(clean_data,
                                     bins=bins,
                                     color='#5B8FA3',  # 低饱和度蓝灰色
                                     edgecolor='#2C3E50',
                                     linewidth=1.2,
                                     alpha=0.7,
                                     density=True,  # 归一化，便于叠加密度曲线
                                     label='Histogram')

    # 绘制核密度估计曲线（KDE）
    kde_data = clean_data.values
    kde = sns.kdeplot(data=kde_data,
                      ax=ax,
                      color='#C0504D',  # 低饱和度红色
                      linewidth=2.5,
                      alpha=0.8,
                      label='KDE')

    # 添加均值线
    mean_val = clean_data.mean()
    ax.axvline(mean_val,
               color='#9BBB59',  # 低饱和度绿色
               linestyle='--',
               linewidth=2,
               alpha=0.8,
               label=f'Mean = {mean_val:.2f}')

    # 添加中位数线
    median_val = clean_data.median()
    ax.axvline(median_val,
               color='#8064A2',  # 低饱和度紫色
               linestyle='-.',
               linewidth=2,
               alpha=0.8,
               label=f'Median = {median_val:.2f}')

    # 设置标题和标签
    # ax.set_title(f'Distribution of {column_name}',
    #              fontsize=14, fontweight='bold', pad=20, color='#2C3E50')
    ax.set_xlabel(column_name, fontsize=12, fontweight='semibold', color='#2C3E50')
    ax.set_ylabel('Density', fontsize=12, fontweight='semibold', color='#2C3E50')

    # 添加网格线
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='#7F8C8D')
    ax.set_axisbelow(True)

    # 优化刻度
    ax.tick_params(axis='both', labelsize=10, colors='#2C3E50')

    # 计算统计信息
    std_val = clean_data.std()
    min_val = clean_data.min()
    max_val = clean_data.max()
    q25 = clean_data.quantile(0.25)
    q75 = clean_data.quantile(0.75)

    # 添加统计信息文本框
    stats_text = (f'N = {len(clean_data)}\n'
                  f'Mean = {mean_val:.2f}\n'
                  f'Std = {std_val:.2f}\n'
                  f'Median = {median_val:.2f}\n'
                  f'Q1 = {q25:.2f}\n'
                  f'Q3 = {q75:.2f}\n'
                  f'Range = [{min_val:.2f}, {max_val:.2f}]')

    ax.text(0.98, 0.97, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white',
                      edgecolor='#BDC3C7', alpha=0.9, linewidth=1),
            family='monospace')  # 使用等宽字体使数字对齐

    # 添加图例
    ax.legend(loc='upper left',
              fontsize=9,
              framealpha=0.9,
              edgecolor='#BDC3C7')

    # 自动调整布局
    plt.tight_layout()

    # 保存图像
    base_name = f'{column_name}_distribution'
    # plt.savefig(os.path.join(target_path, f'{base_name}_con.png'),
    #             dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(target_path, f'{base_name}_con.pdf'),
                dpi=900, bbox_inches='tight', facecolor='white')

    plt.close()

    print(f"连续型数据图表已保存至: {target_path}")
    # print(f"  - {base_name}.png (高分辨率位图)")
    print(f"  - {base_name}.pdf (矢量图)")
    print(f"\n数据统计摘要:")
    print(f"  样本量: {len(clean_data)}")
    print(f"  均值: {mean_val:.3f}")
    print(f"  标准差: {std_val:.3f}")
    print(f"  中位数: {median_val:.3f}")


def plot_battery_timeseries(file_path, target_path, timestamp_col='timestamp'):
    """
    绘制电池相关的三个指标随索引变化的趋势图
    X轴显示5个时间刻度点
    采用3×1的子图布局

    Parameters:
    -----------
    file_path : str
        CSV文件路径
    target_path : str
        保存图片的目标路径
    timestamp_col : str, optional
        时间戳列名，默认为'timestamp'
    """
    # 创建目标文件夹
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # 读取数据
    data = pd.read_csv(file_path)

    # 处理时间戳 - 用于显示在x轴刻度上
    try:
        data[timestamp_col] = pd.to_datetime(data[timestamp_col], format='%Y-%m-%d %H:%M:%S')
    except:
        try:
            data[timestamp_col] = pd.to_datetime(data[timestamp_col])
        except:
            print(f"警告: 无法解析时间戳列 '{timestamp_col}'")

    # 定义要绘制的列和对应的标签
    columns = ['battery_current', 'battery_voltage', 'battery_power']
    ylabels = ['Current (A)', 'Voltage (V)', 'Power (W)']
    colors = ['#5B8FA3', '#C0504D', '#9BBB59']  # 蓝灰、砖红、橄榄绿

    # 设置科研绘图风格
    plt.style.use('seaborn-v0_8-paper')

    # 创建3×1的子图
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), dpi=300, sharex=True)
    # fig.suptitle('Battery Metrics Time Series Analysis',
    #              fontsize=16, fontweight='bold', y=0.995, color='#2C3E50')

    # 计算时间范围用于显示
    time_start = data[timestamp_col].min()
    time_end = data[timestamp_col].max()
    time_range = (time_end - time_start).total_seconds()
    total_points = len(data)
    sampling_rate = total_points / time_range if time_range > 0 else 0

    # 遍历每个子图
    for idx, (ax, col, ylabel, color) in enumerate(zip(axes, columns, ylabels, colors)):
        # 获取数据 - 使用索引作为x轴
        x_data = np.arange(len(data))
        y_data = data[col].values

        # 绘制主折线图
        ax.plot(x_data, y_data,
                color=color,
                linewidth=1.2,
                alpha=0.8,
                label=ylabel,
                rasterized=True)

        # 添加移动平均线（平滑趋势）
        window_size = max(int(len(y_data) * 0.02), 10)
        if len(y_data) >= window_size:
            moving_avg = pd.Series(y_data).rolling(window=window_size, center=True).mean()
            ax.plot(x_data, moving_avg,
                    color='#2C3E50',
                    linewidth=2.5,
                    alpha=0.7,
                    linestyle='--',
                    label=f'Moving Avg (n={window_size})')

        # 添加均值参考线
        mean_val = np.nanmean(y_data)
        ax.axhline(mean_val,
                   color='#E67E22',
                   linestyle=':',
                   linewidth=2,
                   alpha=0.6,
                   label=f'Mean = {mean_val:.2f}')

        # 填充标准差区域
        std_val = np.nanstd(y_data)
        ax.fill_between(x_data,
                        mean_val - std_val,
                        mean_val + std_val,
                        color=color,
                        alpha=0.15,
                        label=f'±1σ = ±{std_val:.2f}')

        # 设置标签
        ax.set_ylabel(ylabel, fontsize=11, fontweight='semibold', color='#2C3E50')

        # 添加网格线
        ax.grid(True, linestyle='--', alpha=0.3, color='#7F8C8D', axis='both')
        ax.set_axisbelow(True)

        # 优化刻度
        ax.tick_params(axis='both', labelsize=9, colors='#2C3E50')

        # 计算统计信息
        min_val = np.nanmin(y_data)
        max_val = np.nanmax(y_data)
        median_val = np.nanmedian(y_data)

        # 添加统计信息文本框
        # stats_text = (f'N = {len(y_data):,}\n'
        #               f'Mean = {mean_val:.2f}\n'
        #               f'Std = {std_val:.2f}\n'
        #               f'Med = {median_val:.2f}\n'
        #               f'Min = {min_val:.2f}\n'
        #               f'Max = {max_val:.2f}')
        #
        # ax.text(0.98, 0.97, stats_text,
        #         transform=ax.transAxes,
        #         fontsize=8,
        #         verticalalignment='top',
        #         horizontalalignment='right',
        #         bbox=dict(boxstyle='round', facecolor='white',
        #                   edgecolor='#BDC3C7', alpha=0.9, linewidth=1),
        #         family='monospace')

        # 添加图例
        ax.legend(loc='upper left',
                  fontsize=8,
                  framealpha=0.9,
                  edgecolor='#BDC3C7',
                  ncol=2)

        # 添加子图标签 (a), (b), (c)
        # ax.text(0.02, 0.97, f'({chr(97 + idx)})',
        #         transform=ax.transAxes,
        #         fontsize=12,
        #         fontweight='bold',
        #         verticalalignment='top',
        #         horizontalalignment='left',
        #         color='#2C3E50',
        #         bbox=dict(boxstyle='round', facecolor='white',
        #                   edgecolor='none', alpha=0.7))

    # 设置最下方子图的x轴标签
    axes[-1].set_xlabel('Time', fontsize=12, fontweight='semibold', color='#2C3E50')

    # 设置x轴刻度：将数据分成5份，显示对应的时间
    num_ticks = 5
    tick_indices = np.linspace(0, len(data) - 1, num_ticks, dtype=int)
    tick_labels = [data[timestamp_col].iloc[i].strftime('%H:%M:%S') for i in tick_indices]

    axes[-1].set_xticks(tick_indices)
    axes[-1].set_xticklabels(tick_labels)

    # 调整子图间距
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # 保存图像
    base_name = 'battery_metrics_timeseries'
    plt.savefig(os.path.join(target_path, f'{base_name}.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(target_path, f'{base_name}.pdf'),
                bbox_inches='tight', facecolor='white')

    plt.close()

    # 打印统计摘要
    print(f"电池指标时间序列图已保存至: {target_path}")
    print(f"  - {base_name}.png (高分辨率位图)")
    print(f"  - {base_name}.pdf (矢量图)")
    print(f"\n时间序列信息:")
    print("=" * 60)
    print(f"  起始时间: {time_start}")
    print(f"  结束时间: {time_end}")
    print(f"  总时长: {time_range:.2f} 秒 ({time_range / 60:.2f} 分钟 / {time_range / 3600:.2f} 小时)")
    print(f"  总数据点: {total_points:,}")
    print(f"  平均采样率: {sampling_rate:.2f} Hz")

    print(f"\nX轴时间刻度:")
    for i, (idx, label) in enumerate(zip(tick_indices, tick_labels)):
        print(f"  刻度 {i + 1}: 索引 {idx} -> {label}")

    print(f"\n数据统计摘要:")
    print("=" * 60)

    for col, ylabel in zip(columns, ylabels):
        y_data = data[col].values
        print(f"\n{ylabel}:")
        print(f"  样本量: {len(y_data):,}")
        print(f"  均值: {np.nanmean(y_data):.3f}")
        print(f"  标准差: {np.nanstd(y_data):.3f}")
        print(f"  中位数: {np.nanmedian(y_data):.3f}")
        print(f"  范围: [{np.nanmin(y_data):.3f}, {np.nanmax(y_data):.3f}]")

    print("=" * 60)


def plot_status_comparison(file_path, target_path):
    """
    绘制mobile_status和wifi_status的对比柱状图
    展示0和1两种状态的分布情况

    Parameters:
    -----------
    file_path : str
        CSV文件路径
    target_path : str
        保存图片的目标路径
    """
    # 创建目标文件夹
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # 读取数据
    data = pd.read_csv(file_path)

    # 设置科研绘图风格
    plt.style.use('seaborn-v0_8-paper')

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

    # 定义状态和标签
    statuses = ['mobile_status', 'wifi_status']
    labels = ['Mobile Status', 'WiFi Status']
    colors = ['#5B8FA3', '#C0504D']  # 蓝灰色和砖红色

    # 统计每个状态的0和1的数量
    status_0_counts = []
    status_1_counts = []

    for status in statuses:
        count_0 = (data[status] == 0).sum()
        count_1 = (data[status] == 1).sum()
        status_0_counts.append(count_0)
        status_1_counts.append(count_1)

    # 设置柱状图的位置
    x = np.arange(len(statuses))
    width = 0.35  # 柱子的宽度

    # 绘制分组柱状图
    bars1 = ax.bar(x - width / 2, status_0_counts, width,
                   label='Status = 0 (Inactive)',
                   color='#7F8C8D',  # 灰色表示关闭/未激活
                   edgecolor='#2C3E50',
                   linewidth=1.2,
                   alpha=0.85)

    bars2 = ax.bar(x + width / 2, status_1_counts, width,
                   label='Status = 1 (Active)',
                   color='#27AE60',  # 绿色表示开启/激活
                   edgecolor='#2C3E50',
                   linewidth=1.2,
                   alpha=0.85)

    # 在柱状图上方添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            # 计算百分比
            total = sum(status_0_counts[0] + status_1_counts[0] if bar in bars1[:1] or bar in bars2[:1]
                        else status_0_counts[1] + status_1_counts[1] for _ in range(1))

            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{int(height)}\n({height / len(data) * 100:.1f}%)',
                    ha='center', va='bottom',
                    fontsize=9, color='#2C3E50',
                    fontweight='semibold')

    # 设置标题和标签
    # ax.set_title('Comparison of Mobile and WiFi Status Distribution',
    #              fontsize=14, fontweight='bold', pad=20, color='#2C3E50')
    ax.set_ylabel('Frequency (Count)', fontsize=12, fontweight='semibold', color='#2C3E50')
    ax.set_xlabel('Status Type', fontsize=12, fontweight='semibold', color='#2C3E50')

    # 设置x轴刻度
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)

    # 添加网格线 - 仅水平方向
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='#7F8C8D')
    ax.set_axisbelow(True)

    # 优化刻度
    ax.tick_params(axis='both', labelsize=10, colors='#2C3E50')

    # 添加图例
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.65, 0.98),  # x=0.75 表示从左到右 3/4，y=0.98 靠近顶部并避免被裁剪
        bbox_transform=ax.transAxes,  # 使用轴坐标系
        fontsize=10,
        framealpha=0.9,
        edgecolor='#BDC3C7',
        title='Status Value',
        title_fontsize=10
    )

    # 自动调整y轴范围，留出空间给标签
    y_max = max(max(status_0_counts), max(status_1_counts))
    ax.set_ylim(0, y_max * 1.15)

    # 调整布局
    plt.tight_layout()

    # 保存图像
    base_name = 'status_comparison'
    plt.savefig(os.path.join(target_path, f'{base_name}.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(target_path, f'{base_name}.pdf'),
                bbox_inches='tight', facecolor='white')

    plt.close()


def plot_correlation_heatmap(file_path, target_path):
    """
    绘制多个变量之间的相关性热力图
    包括：bright_level, gps_status, battery_level, battery_current,
          battery_voltage, battery_power, mobile_status, wifi_status

    Parameters:
    -----------
    file_path : str
        CSV文件路径
    target_path : str
        保存图片的目标路径
    """
    # 创建目标文件夹
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # 读取数据
    data = pd.read_csv(file_path)

    # 定义要分析的列
    columns = ['screen_status', 'gps_status', 'battery_level',
               'battery_current', 'battery_voltage', 'battery_power',
               'mobile_status', 'wifi_status']

    # 检查列是否存在
    available_columns = [col for col in columns if col in data.columns]
    if len(available_columns) < len(columns):
        missing = set(columns) - set(available_columns)
        print(f"警告: 以下列不存在: {missing}")

    # 提取相关列的数据
    subset_data = data[available_columns]

    # 计算相关性矩阵
    correlation_matrix = subset_data.corr()

    # 设置科研绘图风格
    plt.style.use('seaborn-v0_8-paper')

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)

    # 使用科研配色方案 - RdBu_r (红蓝配色，中心为白色)
    # 这是科研论文中最常用的相关性热力图配色之一
    cmap = plt.cm.RdBu_r  # 红色表示负相关，蓝色表示正相关

    # 绘制热力图
    im = ax.imshow(correlation_matrix, cmap=cmap, aspect='auto',
                   vmin=-1, vmax=1, interpolation='nearest')

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # cbar.set_label('Correlation Coefficient',
    #                rotation=270, labelpad=25,
    #                fontsize=11, fontweight='semibold', color='#2C3E50')
    cbar.ax.tick_params(labelsize=9, colors='#2C3E50')

    # 设置刻度
    ax.set_xticks(np.arange(len(available_columns)))
    ax.set_yticks(np.arange(len(available_columns)))

    # 设置刻度标签 - 优化显示名称
    label_mapping = {
        'screen_status': 'Brightness',
        'gps_status': 'GPS',
        'battery_level': 'Battery Level',
        'battery_current': 'Current',
        'battery_voltage': 'Voltage',
        'battery_power': 'Power',
        'mobile_status': 'Mobile',
        'wifi_status': 'WiFi'
    }

    labels = [label_mapping.get(col, col) for col in available_columns]

    ax.set_xticklabels(labels, fontsize=10, color='#2C3E50')
    ax.set_yticklabels(labels, fontsize=10, color='#2C3E50')

    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # 在每个格子中添加相关系数数值
    for i in range(len(available_columns)):
        for j in range(len(available_columns)):
            value = correlation_matrix.iloc[i, j]
            # 根据背景颜色选择文字颜色（深色背景用白色文字，浅色背景用黑色文字）
            text_color = 'white' if abs(value) > 0.5 else '#2C3E50'
            text = ax.text(j, i, f'{value:.2f}',
                           ha='center', va='center',
                           color=text_color, fontsize=9,
                           fontweight='semibold')

    # 添加网格线分隔每个单元格
    ax.set_xticks(np.arange(len(available_columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(available_columns)) - 0.5, minor=True)
    ax.grid(which='minor', color='#34495E', linestyle='-', linewidth=1.5)
    ax.tick_params(which='minor', size=0)

    # 设置标题
    # ax.set_title('Correlation Matrix of Battery and Network Metrics',
    #              fontsize=14, fontweight='bold', pad=20, color='#2C3E50')

    # 添加统计信息文本框
    # 找出最强的正相关和负相关（排除对角线）
    corr_no_diag = correlation_matrix.copy()
    np.fill_diagonal(corr_no_diag.values, np.nan)

    # 找到最大正相关
    max_corr = corr_no_diag.max().max()
    max_corr_idx = np.where(corr_no_diag == max_corr)
    if len(max_corr_idx[0]) > 0:
        max_pair = (available_columns[max_corr_idx[0][0]],
                    available_columns[max_corr_idx[1][0]])
    else:
        max_pair = ('N/A', 'N/A')

    # 找到最大负相关
    min_corr = corr_no_diag.min().min()
    min_corr_idx = np.where(corr_no_diag == min_corr)
    if len(min_corr_idx[0]) > 0:
        min_pair = (available_columns[min_corr_idx[0][0]],
                    available_columns[min_corr_idx[1][0]])
    else:
        min_pair = ('N/A', 'N/A')

    # stats_text = (f'Sample Size: {len(data):,}\n'
    #               f'Variables: {len(available_columns)}\n\n'
    #               f'Strongest Positive:\n'
    #               f'{label_mapping.get(max_pair[0], max_pair[0])} ↔\n'
    #               f'{label_mapping.get(max_pair[1], max_pair[1])}\n'
    #               f'r = {max_corr:.3f}\n\n'
    #               f'Strongest Negative:\n'
    #               f'{label_mapping.get(min_pair[0], min_pair[0])} ↔\n'
    #               f'{label_mapping.get(min_pair[1], min_pair[1])}\n'
    #               f'r = {min_corr:.3f}')
    #
    # ax.text(1.15, 0.5, stats_text,
    #         transform=ax.transAxes,
    #         fontsize=9,
    #         verticalalignment='center',
    #         horizontalalignment='left',
    #         bbox=dict(boxstyle='round', facecolor='white',
    #                   edgecolor='#BDC3C7', alpha=0.9, linewidth=1),
    #         family='monospace')

    # 调整布局
    plt.tight_layout()

    # 保存图像
    base_name = 'correlation_heatmap'
    plt.savefig(os.path.join(target_path, f'{base_name}.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(target_path, f'{base_name}.pdf'),
                bbox_inches='tight', facecolor='white')

    plt.close()

    # 打印统计摘要
    print(f"相关性热力图已保存至: {target_path}")
    print(f"  - {base_name}.png (高分辨率位图)")
    print(f"  - {base_name}.pdf (矢量图)")
    print(f"\n相关性分析摘要:")
    print("=" * 60)
    print(f"样本数量: {len(data):,}")
    print(f"分析变量数: {len(available_columns)}")
    print(f"\n最强正相关:")
    print(f"  {max_pair[0]} ↔ {max_pair[1]}")
    print(f"  相关系数: {max_corr:.4f}")
    print(f"\n最强负相关:")
    print(f"  {min_pair[0]} ↔ {min_pair[1]}")
    print(f"  相关系数: {min_corr:.4f}")
    print("\n相关性矩阵:")
    print(correlation_matrix.round(3))
    print("=" * 60)
if __name__ == "__main__":
    file_path = r'./A dataset from the daily use of features in Android devices/Dynamic data/20220412/72deadc7f8553ef1/72deadc7f8553ef1_20220412_dynamic_processed.csv'  # 替换为你的数据文件路径
    column_name = ('wifi_status')  # 替换为你要绘制分布图的列名
    target_path = './result/single_file_analyze'  # 替换为你想保存图像的文件夹路径
    # plot_continuous(file_path, column_name, target_path)
    # plot_status_comparison(file_path, target_path)
    plot_correlation_heatmap(file_path, target_path)