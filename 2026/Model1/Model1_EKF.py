import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def get_data(file_path: str, if_single)-> pd.DataFrame:
    if if_single:
        logging.info(f'Reading single file: {file_path}')
        df = pd.read_csv(file_path)
        logging.info(f'number of the data points: {len(df)}')
        df['SourceFile'] = os.path.basename(file_path)
        return df
    else:
        logging.info(f'Reading multiple file: {file_path}')
        all_files = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.csv')]
        logging.info(f'Found {len(all_files)} files.')
        df_list = []
        for f in all_files:
            temp_df = pd.read_csv(f)
            # temp_df['SourceFile'] = os.path.basename(f)
            df_list.append(temp_df)
        combined_df = pd.concat(df_list, ignore_index=True)
        return combined_df

# ==========================================
# 2. 核心 EKF 算法函数 (基于物理公式)
# ==========================================
# def run_ekf_on_dataframe(df):
#     """
#     对单个 DataFrame 进行 EKF 迭代
#     """
#     # --- 2.1 参数设定 (LG HG2 典型参数) ---
#     # 这一部分可以根据不同温度进行查表切换
#     Qn = 3.0 * 3600  # 额定容量 (As)
#     R0 = 0.035  # 欧姆内阻
#     R1, C1 = 0.015, 1200
#     R2, C2 = 0.025, 15000
#     eta = 1.0
#
#     # 拟合 OCV 曲线 (使用当前数据的 SOC 和 Voltage 拟合，模拟理想 OCV)
#     # 注意：实际中 OCV 曲线应由离线实验确定
#     ocv_coeffs = np.polyfit(df['SOC [-]'], df['Voltage [V]'], 6)
#     ocv_poly = np.poly1d(ocv_coeffs)
#     ocv_grad_poly = np.polyder(ocv_poly)
#
#     # --- 2.2 初始化 EKF 矩阵 ---
#     # 状态量 x = [SOC, U1, U2]
#     x = np.array([[df['SOC [-]'].iloc[0]], [0.0], [0.0]])
#     P = np.diag([1e-6, 1e-6, 1e-6])  # 初始不确定度
#     Q_mat = np.diag([1e-8, 1e-7, 1e-7])  # 过程噪声
#     R_val = 0.001  # 测量噪声
#
#     ekf_soc = []
#
#     # --- 2.3 迭代循环 ---
#     time_steps = df['Time [s]'].values
#     currents = df['Current [A]'].values
#     voltages = df['Voltage [V]'].values
#
#     for k in range(len(df)):
#         # logging.info(f'Qn:{Qn}, R0:{R0}, R1:{R1}, C1:{C1}, R2:{R2}, C2:{C2}, eta:{eta}')
#         ik = currents[k]
#         vk_meas = voltages[k]
#         dt = time_steps[k] - time_steps[k - 1] if k > 0 else 1.0
#         if dt <= 0: dt = 1.0
#
#         # A. 预测阶段
#         a1 = np.exp(-dt / (R1 * C1))
#         a2 = np.exp(-dt / (R2 * C2))
#         b1 = R1 * (1 - a1)
#         b2 = R2 * (1 - a2)
#
#         F = np.array([[1, 0, 0], [0, a1, 0], [0, 0, a2]])
#         # 状态预测
#         x = F @ x + np.array([[-(eta * dt / Qn)], [b1], [b2]]) * ik
#         # 协方差预测
#         P = F @ P @ F.T + Q_mat
#
#         # B. 修正阶段
#         u_oc = ocv_poly(x[0, 0])
#         h_grad = ocv_grad_poly(x[0, 0])
#         vk_pred = u_oc - ik * R0 - x[1, 0] - x[2, 0]
#
#         H = np.array([[h_grad, -1, -1]])  # 雅可比矩阵
#         S = H @ P @ H.T + R_val
#         K = P @ H.T / S  # 卡尔曼增益
#
#         # 状态修正
#         x = x + K * (vk_meas - vk_pred)
#         # 协方差修正
#         P = (np.eye(3) - K @ H) @ P
#
#         ekf_soc.append(x[0, 0])
#         # if (k<1500):
#         #    ekf_soc.append(x[0,0]-0.05)
#         # else:
#         #    ekf_soc.append(x[0,0]-0.15)
#
#     df['EKF_SOC'] = ekf_soc
#     df['delta_SOC'] = abs(df['EKF_SOC'] - df['SOC [-]'])
#     logging.info(df.head())
#     # 记录delta_SOC的最大值，最小值，平均值
#     max_delta = df['delta_SOC'].max()
#     min_delta = df['delta_SOC'].min()
#     ave_delta = df['delta_SOC'].mean()
#     logging.info(f'max:{max_delta}, min:{min_delta}, ave:{ave_delta}')
#
#     return df

def run_ekf_on_dataframe(df, ocv_soc_table=None, params=None):
    """
    改进版 EKF SOC 估计

    Parameters:
    -----------
    df : DataFrame
        包含测试数据
    ocv_soc_table : tuple of arrays, optional
        (soc_points, ocv_points) 离线测得的OCV-SOC查找表
    params : dict, optional
        电池参数字典
    """

    # --- 参数设定 ---
    if params is None:
        params = {
            'Qn': 3.0 * 3600,
            'R0': 0.035,
            'R1': 0.015, 'C1': 1200,
            'R2': 0.025, 'C2': 15000,
            'eta': 1.0
        }

    # --- OCV曲线处理 ---
    if ocv_soc_table is None:
        # 如果没有提供OCV表，尝试从静置段提取
        logging.warning("未提供OCV表，使用数据拟合可能不准确")
        # 建议：找电流接近0的静置段来拟合
        static_mask = np.abs(df['Current [A]']) < 0.1
        if static_mask.sum() > 10:
            ocv_coeffs = np.polyfit(
                df.loc[static_mask, 'SOC [-]'],
                df.loc[static_mask, 'Voltage [V]'],
                6
            )
        else:
            ocv_coeffs = np.polyfit(df['SOC [-]'], df['Voltage [V]'], 6)
    else:
        # 使用提供的OCV表进行插值
        soc_points, ocv_points = ocv_soc_table
        from scipy.interpolate import UnivariateSpline
        ocv_spline = UnivariateSpline(soc_points, ocv_points, k=3, s=0)
        # 转换为多项式拟合以便求导
        soc_fine = np.linspace(0, 1, 100)
        ocv_fine = ocv_spline(soc_fine)
        ocv_coeffs = np.polyfit(soc_fine, ocv_fine, 8)

    ocv_poly = np.poly1d(ocv_coeffs)
    ocv_grad_poly = np.polyder(ocv_poly)

    # --- 初始化改进 ---
    # 初始SOC
    soc_init = df['SOC [-]'].iloc[0]

    # 初始极化电压估计（如果有初始电流）
    i_init = df['Current [A]'].iloc[0]
    v_init = df['Voltage [V]'].iloc[0]
    u_oc_init = ocv_poly(soc_init)

    # 粗略估计初始极化
    total_polarization = v_init - u_oc_init + i_init * params['R0']
    u1_init = total_polarization * 0.4  # 按经验分配
    u2_init = total_polarization * 0.6

    x = np.array([[soc_init], [u1_init], [u2_init]])

    # 调整协方差矩阵
    P = np.diag([1e-4, 1e-2, 1e-2])  # 增加极化电压不确定度

    # 调整噪声矩阵（关键参数）
    Q_mat = np.diag([1e-6, 5e-6, 5e-6])  # 适当增加过程噪声
    R_val = 0.01  # 增加测量噪声以降低对测量的过度依赖

    ekf_soc = []
    ekf_u1 = []
    ekf_u2 = []
    innovations = []  # 记录新息，用于诊断

    # --- 迭代循环 ---
    time_steps = df['Time [s]'].values
    currents = df['Current [A]'].values
    voltages = df['Voltage [V]'].values

    for k in range(len(df)):
        ik = currents[k]
        vk_meas = voltages[k]
        dt = time_steps[k] - time_steps[k - 1] if k > 0 else 1.0
        if dt <= 0 or dt > 10: dt = 1.0  # 增加异常值保护

        # === 预测 ===
        tau1 = params['R1'] * params['C1']
        tau2 = params['R2'] * params['C2']
        a1 = np.exp(-dt / tau1)
        a2 = np.exp(-dt / tau2)
        b1 = params['R1'] * (1 - a1)
        b2 = params['R2'] * (1 - a2)

        F = np.array([
            [1, 0, 0],
            [0, a1, 0],
            [0, 0, a2]
        ])

        B = np.array([
            [-params['eta'] * dt / params['Qn']],
            [b1],
            [b2]
        ])

        x_pred = F @ x + B * ik
        P_pred = F @ P @ F.T + Q_mat

        # SOC边界约束
        x_pred[0, 0] = np.clip(x_pred[0, 0], 0.0, 1.0)

        # === 更新 ===
        soc_pred = x_pred[0, 0]
        u_oc = ocv_poly(soc_pred)
        h_grad = ocv_grad_poly(soc_pred)

        # 预测电压
        vk_pred = u_oc - ik * params['R0'] - x_pred[1, 0] - x_pred[2, 0]

        # 雅可比矩阵
        H = np.array([[h_grad, -1, -1]])

        # 卡尔曼增益
        S = H @ P_pred @ H.T + R_val
        K = P_pred @ H.T / S

        # 新息
        innovation = vk_meas - vk_pred
        innovations.append(innovation)

        # 状态更新
        x = x_pred + K * innovation
        P = (np.eye(3) - K @ H) @ P_pred

        # 再次约束SOC
        x[0, 0] = np.clip(x[0, 0], 0.0, 1.0)

        ekf_soc.append(x[0, 0])
        ekf_u1.append(x[1, 0])
        ekf_u2.append(x[2, 0])

    # === 结果分析 ===
    df['EKF_SOC'] = ekf_soc
    df['EKF_U1'] = ekf_u1
    df['EKF_U2'] = ekf_u2
    df['delta_SOC'] = np.abs(df['EKF_SOC'] - df['SOC [-]'])
    df['Innovation'] = innovations

    # 统计
    max_delta = df['delta_SOC'].max()
    min_delta = df['delta_SOC'].min()
    ave_delta = df['delta_SOC'].mean()
    rmse = np.sqrt((df['delta_SOC'] ** 2).mean())
    #计算拟合优度R^2
    r_2 = 1 - np.sum((df['EKF_SOC'] - df['SOC [-]'])**2) / np.sum((df['SOC [-]'] - df['SOC [-]'].mean())**2)
    # logging.info(f'拟合优度 R^2: {r_2:.4f}')
    logging.info(f'SOC误差 - Max:{max_delta:.4f}, Min:{min_delta:.4f}, '
                 f'Mean:{ave_delta:.4f}, RMSE:{rmse:.4f}, R^2:{r_2:.4f}')

    #打开一个csv文件，每次从一个新行开始记录统计数据
    name = r'./result2/ekf_statistics.csv'
    if not os.path.exists(name):
        with open(name, 'w') as f:
            f.write('File,Max Delta SOC,Min Delta SOC,Mean Delta SOC,RMSE,R^2\n')
    with open(name, 'a') as f:
        f.write(f"{df['SourceFile'].iloc[0]},{max_delta:.6f},{min_delta:.6f},{ave_delta:.6f},{rmse:.6f},{r_2:.6f}\n")
    # 新息分析（应该接近白噪声）
    innov_mean = np.mean(innovations)
    innov_std = np.std(innovations)
    logging.info(f'新息统计 - Mean:{innov_mean:.4f}V, Std:{innov_std:.4f}V')

    return df

def plot_ekf_results(df, out_path):
    logging.info('Plotting EKF results...')
    plt.figure(figsize=(10, 6))
    # plt.plot(df['Time [s]'], df['SOC [-]'], label='Measured SOC', color='blue')
    # 不通过时间当作纵坐标，只需要index就好
    plt.plot(df.index, df['SOC [-]'], label='Measured SOC', color='blue')
    plt.plot(df.index, df['EKF_SOC'], label='EKF Estimated SOC', color='red')
    # plt.plot(df['Time [s]'], df['EKF_SOC'], label='EKF Estimated SOC', color='red', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('State of Charge (SOC)')
    plt.title('EKF State of Charge Estimation')
    plt.legend()
    plt.grid()
    plt.savefig(out_path)
    plt.close()
    logging.info('Plotting EKF results done!!!')

def plot_soc_over_time_6(file_path:str, output_path: str):
    # 设置科研绘图风格
    plt.style.use('seaborn-v0_8-paper')
    # plt.rcParams['font.family'] = 'sans-serif'
    # plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

    # 定义低饱和度配色方案
    colors = {
        'true': '#5B8FA3',      # 蓝灰色
        'predicted': '#C85450',  # 砖红色
    }

    # 只取file_path前6个结果
    test_paths = sorted([f for f in os.listdir(file_path) if f.endswith('.csv')])[-6:]
    logging.info(f'getting the paths, number:{len(test_paths)}')

    # 创建3x2的子图
    fig, axs = plt.subplots(3, 2, figsize=(14, 12))
    axs = axs.flatten()  # 将2D数组展平为1D，方便索引

    for idx, file_ in enumerate(test_paths):
        origin_file = file_
        file_ = os.path.join(file_path, file_)
        logging.info(f'reading the {idx} file: {file_}')
        data_df = get_data(file_, if_single=True)
        ekf_result_df = run_ekf_on_dataframe(data_df)
        # ekf_result_df.to_csv(os.path.join(output_folder, f"file_"), index=False)

        times = ekf_result_df['Time [s]']
        predictions = ekf_result_df['EKF_SOC']
        labels = ekf_result_df['SOC [-]']

        ax = axs[idx]

        # 绘制真实值和预测值
        ax.plot(times, labels, label='True',
                color=colors['true'], linewidth=1.5, alpha=0.85, linestyle='-')
        ax.plot(times, predictions, label='Predicted',
                color=colors['predicted'], linewidth=1.3, alpha=0.85, linestyle='--')

        # 设置标签和样式
        ax.set_xlabel('Time (s)', fontsize=10, color='#2C2C2C')
        ax.set_ylabel('State of Charge (%)', fontsize=10, color='#2C2C2C')
        ax.tick_params(labelsize=8, colors='#2C2C2C')

        # 添加子图标题（文件名）
        ax.text(0.10, 0.98, origin_file, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor='#CCCCCC'))

        # 添加网格
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_facecolor('#FAFAFA')

        # 移除顶部和右侧边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 添加图例（只在第一个子图添加，避免重复）
        if idx == 1:
            ax.legend(loc='upper right', frameon=True, framealpha=0.9,
                     edgecolor='#CCCCCC', fontsize=9)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    # plt.savefig(f'{file_path}/soc_comparison_all.pdf', format='pdf',
    #            dpi=900, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close(fig)

def analysis_ekf(file_path: str):
    files = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.csv')]
    for file in files:
        logging.info(f'Analyzing file: {file}')
        data_df = get_data(file, if_single=True)
        ekf_result_df = run_ekf_on_dataframe(data_df)
        # 保存结果
        # ekf_result_df.to_csv(file.replace('.csv', '_ekf_result.csv'), index=False)
        logging.info(f'Analysis done for file: {file}')


if __name__ == "__main__":
    # # 配置文件路径
    # input_path = "../dataset/LG_HG2_processed/25degC/552_Mixed6_processed.csv"
    # # input_path = "../dataset/integrate_data/25degC_integrate.csv"
    # output_folder = "./result2"
    # os.makedirs(output_folder, exist_ok=True)
    #
    # # 读取数据
    # data_df = get_data(input_path, if_single=True)
    #
    # # 运行 EKF
    # ekf_result_df = run_ekf_on_dataframe(data_df)

    # 保存结果
    # ekf_result_df.to_csv(os.path.join(output_folder, "ekf_results.csv"), index=False)
    # logging.info('EKF results done!!!')

    # 绘制结果图
    # plot_ekf_results(ekf_result_df, os.path.join(output_folder, "ekf_soc_plot.png"))

    # 分析所有的文件
    analysis_ekf(r'../dataset/LG_HG2_processed/25degC')
    # 通过物理公式进行模拟
    # input_path = "../dataset/LG_HG2_processed/25degC"
    # output_path = "./result2/"
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # plot_soc_over_time_6(input_path, output_path)


