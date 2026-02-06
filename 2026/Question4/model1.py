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
    name = r'./result/ekf_statistics.csv'
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
    # # # 配置文件路径
    # input_path = "../dataset/LG_HG2_processed/25degC/552_Mixed6_processed.csv"
    # # input_path = "../dataset/integrate_data/25degC_integrate.csv"
    # output_folder = "./result"
    # os.makedirs(output_folder, exist_ok=True)
    # # 读取数据
    # data_df = get_data(input_path, if_single=True)
    #
    # # 运行 EKF
    # ekf_result_df = run_ekf_on_dataframe(data_df)

    analysis_ekf(r'../dataset/LG_HG2_processed/25degC')



