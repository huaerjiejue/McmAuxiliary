# python
import os
import logging
from typing import List, Dict, Tuple, Optional
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无显示环境也能保存图片；如果在本地想直接显示可以改为默认后端
import matplotlib.pyplot as plt
import re

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# 常见列名候选
_TIME_CANDS = ['time', 'time_s', 'time(s)', 'prog time', 'prog_time', 'time stamp', 'timestamp', 'prog']
_VOLT_CANDS = ['voltage', 'voltage(V)', 'v', 'Voltage [V]']
_CURR_CANDS = ['current', 'current(A)', 'i', 'current(A)', 'Current [A]']
_CAP_CANDS = ['capacity', 'capacity(Ah)', 'cap', 'Capacity [Ah]']
_TEMPERATE_CANDS = ['temperature', 'temp', 'temperature(C)', 'Temperature [degC]']
_SOC_CANDS = ['soc', 'state of charge', 'soc(%)', 'SoC [%]', 'SOC [-]']

def integrate_data(file_path: str, target_path: str, keyword:str):
    # 读取file_path中特定的问价据，进行积分处理后保存到target_path
    files = os.listdir(file_path)
    files_to_process = []
    for f in files:
        # 如果f中包含关键字，则整合在一起
        if keyword in f:
            files_to_process.append(f)
    # 把所有需要处理的文件整合在一起
    all_data = []
    for f in files_to_process:
        full_path = os.path.join(file_path, f)
        try:
            df = pd.read_csv(full_path)
            # 找到df中列名称为Timestamp的列
            if 'Timestamp' in df.columns:
                df.rename(columns={'Timestamp': 'Daytime'}, inplace=True)
                df['Daytime'] = pd.to_datetime(df['Daytime']).dt.date
            all_data.append(df)
            logging.info(f'Find file in {full_path}')
        except Exception as e:
            logging.error(f'Failed to read {full_path}: {e}')
    # save the data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        # save, if target_path's directory not exist, create it, if exist, overwrite it
        target_dir = os.path.dirname(target_path)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        combined_df.to_csv(target_path, index=False)
        logging.info(f'Saved integrated data to {target_path}')
    else:
        logging.warning('No data files were processed.')

def analyze_file(file_path: str) -> Optional[Dict[str, pd.Series]]:
    """分析单个文件，提取时间、电压、电流、容量数据"""
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logging.error(f'Failed to read {file_path}: {e}')
        return None

    def find_column(candidates: List[str]) -> Optional[str]:
        for cand in candidates:
            for col in df.columns:
                if re.search(cand, col, re.IGNORECASE):
                    return col
        return None

    # 时间一般内容2018-10-29 14:41:07，获取到每日
    time_col = find_column(_TIME_CANDS)
    volt_col = find_column(_VOLT_CANDS)
    curr_col = find_column(_CURR_CANDS)
    cap_col = find_column(_CAP_CANDS)
    temperature_col = find_column(_TEMPERATE_CANDS)
    soc_col = find_column(_SOC_CANDS)
    if not time_col or not volt_col or not curr_col:
        logging.error(f'Missing required columns in {file_path}')
        return None
    data = {'time': pd.to_datetime(df[time_col]).dt.date,
            'voltage': df[volt_col],
            'current': df[curr_col],
            'temperature': df[temperature_col],
            'soc': df[soc_col]}
    return data


def plot(data: Dict[str, pd.Series], out_path: str) -> bool:
    """绘制时间-电压、电流、容量曲线图"""
    # 设置科研绘图风格
    plt.style.use('seaborn-v0_8-paper')

    # 定义低饱和度配色方案
    colors = {
        'box': '#7C9EB2',  # 蓝灰色
        'median': '#C85450',  # 砖红色
        'whisker': '#5A5A5A',  # 深灰色
    }

    # 时间只需要精确到日就好了，绘制箱线图
    # 分四个子图,分别绘制随时间变换电压,电流,温度,soc值的分布图
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # 获取时间标签
    time_labels = [str(day) for day in sorted(data['time'].unique())]

    # 电压箱线图
    bp1 = axs[0, 0].boxplot(
        [data['voltage'][data['time'] == day] for day in sorted(data['time'].unique())],
        labels=time_labels,
        patch_artist=True,
        boxprops=dict(facecolor=colors['box'], color='#404040', linewidth=1.2, alpha=0.7),
        whiskerprops=dict(color=colors['whisker'], linewidth=1.2),
        capprops=dict(color=colors['whisker'], linewidth=1.2),
        medianprops=dict(color=colors['median'], linewidth=1.5),
        flierprops=dict(marker='o', markerfacecolor='none', markersize=4,
                        markeredgecolor=colors['whisker'], alpha=0.5)
    )
    axs[0, 0].set_ylabel('Voltage (V)', fontsize=11, color='#2C2C2C')
    axs[0, 0].tick_params(labelsize=9, colors='#2C2C2C')
    axs[0, 0].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    axs[0, 0].set_facecolor('#FAFAFA')

    # 电流箱线图
    bp2 = axs[0, 1].boxplot(
        [data['current'][data['time'] == day] for day in sorted(data['time'].unique())],
        labels=time_labels,
        patch_artist=True,
        boxprops=dict(facecolor=colors['box'], color='#404040', linewidth=1.2, alpha=0.7),
        whiskerprops=dict(color=colors['whisker'], linewidth=1.2),
        capprops=dict(color=colors['whisker'], linewidth=1.2),
        medianprops=dict(color=colors['median'], linewidth=1.5),
        flierprops=dict(marker='o', markerfacecolor='none', markersize=4,
                        markeredgecolor=colors['whisker'], alpha=0.5)
    )
    axs[0, 1].set_ylabel('Current (A)', fontsize=11, color='#2C2C2C')
    axs[0, 1].tick_params(labelsize=9, colors='#2C2C2C')
    axs[0, 1].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    axs[0, 1].set_facecolor('#FAFAFA')

    # 温度箱线图
    if 'temperature' in data and data['temperature'].notnull().any():
        bp3 = axs[1, 0].boxplot(
            [data['temperature'][data['time'] == day] for day in sorted(data['time'].unique())],
            labels=time_labels,
            patch_artist=True,
            boxprops=dict(facecolor=colors['box'], color='#404040', linewidth=1.2, alpha=0.7),
            whiskerprops=dict(color=colors['whisker'], linewidth=1.2),
            capprops=dict(color=colors['whisker'], linewidth=1.2),
            medianprops=dict(color=colors['median'], linewidth=1.5),
            flierprops=dict(marker='o', markerfacecolor='none', markersize=4,
                            markeredgecolor=colors['whisker'], alpha=0.5)
        )
        axs[1, 0].set_ylabel('Temperature (°C)', fontsize=11, color='#2C2C2C')
        axs[1, 0].tick_params(labelsize=9, colors='#2C2C2C')
        axs[1, 0].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    else:
        axs[1, 0].text(0.5, 0.5, 'No Temperature Data', ha='center', va='center',
                       fontsize=11, color='#5A5A5A')
    axs[1, 0].set_facecolor('#FAFAFA')
    axs[1, 0].set_xlabel('Time', fontsize=11, color='#2C2C2C')

    # SoC箱线图
    if 'soc' in data and data['soc'].notnull().any():
        bp4 = axs[1, 1].boxplot(
            [data['soc'][data['time'] == day] for day in sorted(data['time'].unique())],
            labels=time_labels,
            patch_artist=True,
            boxprops=dict(facecolor=colors['box'], color='#404040', linewidth=1.2, alpha=0.7),
            whiskerprops=dict(color=colors['whisker'], linewidth=1.2),
            capprops=dict(color=colors['whisker'], linewidth=1.2),
            medianprops=dict(color=colors['median'], linewidth=1.5),
            flierprops=dict(marker='o', markerfacecolor='none', markersize=4,
                            markeredgecolor=colors['whisker'], alpha=0.5)
        )
        axs[1, 1].set_ylabel('State of Charge (%)', fontsize=11, color='#2C2C2C')
        axs[1, 1].tick_params(labelsize=9, colors='#2C2C2C')
        axs[1, 1].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    else:
        axs[1, 1].text(0.5, 0.5, 'No SoC Data', ha='center', va='center',
                       fontsize=11, color='#5A5A5A')
    axs[1, 1].set_facecolor('#FAFAFA')
    axs[1, 1].set_xlabel('Time', fontsize=11, color='#2C2C2C')

    # 调整布局
    plt.tight_layout()

    # 保存图片,使用更高的DPI以获得更好的质量
    plt.savefig(out_path, dpi=3900, bbox_inches='tight', facecolor='white')
    logging.info(f'Saved plot to {out_path}')
    plt.close()

    return True

def analyze_data(file_path: str, out_dir: Optional[str]=None, show: bool=False) -> bool:
    """分析单个文件并绘制图像"""
    data = analyze_file(file_path)
    if not data:
        return False
    # 生成输出路径
    base_name = os.path.basename(file_path)
    name_no_ext = os.path.splitext(base_name)[0]
    if out_dir:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = os.path.join(out_dir, f'{name_no_ext}_plot.png')
    else:
        out_path = f'{name_no_ext}_plot.png'
    success = plot(data, out_path)
    if show and success:
        img = plt.imread(out_path)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    return success

if __name__ == '__main__':
    # If run from data_process, resolve 25degC under dataset
    default_temp = '25degC'
    total, success = read_and_plot_processed_plots(default_temp, out_dir=None, show=False)
    print(f'Done. total={total}, success_count={len(success)}')
    for p in success:
        print(f'  - {p}')

    # 整合数据，并绘制图像
    file_path = r'../dataset/LG_HG2_processed/25degC'
    target_path = r'../dataset/integrate_data/25degC_integrate.csv'
    # integrate_data(file_path, target_path, 'Mix')
    data = analyze_file(target_path)
    if data:
        plot(data, r'../dataset/integrate_data/25degC_plot.png.pdf')

    #对整合的数据进行数据分析

