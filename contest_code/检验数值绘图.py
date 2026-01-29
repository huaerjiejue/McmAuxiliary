# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from scipy.stats import gaussian_kde
from MathModels.Plot.styles import mp_seaborn_light

# ====================== Global Settings ======================
plt.style.use(mp_seaborn_light())
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 10,
    'figure.dpi': 300,
    'savefig.format': 'pdf',
    'axes.unicode_minus': False
})
# Set English font (adjust path according to your system)
# font = FontProperties(fname='arial.ttf', size=10)  # Example for Windows
# font = FontProperties(fname='/Library/Fonts/Arial.ttf', size=10)  # Example for Mac

# ====================== Data Loading ======================
# Time stability data
time_stability = pd.read_csv('time_stability.csv')
# Parameter sensitivity data
parameter_sens = pd.read_csv('parameter_perturbation.csv')
# Normality test data
normality_test = pd.read_csv('normality_test.csv')

# ====================== Figure 1: Wear Evolution ======================
plt.figure(figsize=(8, 5))

# Plot mean curve
plt.plot(time_stability['time'], time_stability['mean'],
         color='#E74C3C', linewidth=1.5, label='Average Wear')

# Plot confidence interval
plt.fill_between(time_stability['time'],
                 time_stability['ci_lower'],
                 time_stability['ci_upper'],
                 color='#3498DB', alpha=0.3, label='95% Confidence Interval')

# Mark maintenance events
repair_times = [6, 18]
for t in repair_times:
    plt.axvline(t, color='#2ECC71', linestyle='--', linewidth=1, alpha=0.8)
    plt.text(t+0.2, 0.8*max(time_stability['mean']), f'Maintenance ({t}h)',
             rotation=90)

# Styling
plt.title('Ground Wear Evolution Over Time', pad=15)
plt.xlabel('Time (hours)')
plt.ylabel('Cumulative Wear Q')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure1_wear_evolution.pdf')

# ====================== Figure 2: Parameter Sensitivity Radar Chart ======================
# Data preparation
labels = parameter_sens['parameter'].values
st_values = parameter_sens['sensitivity_index'].values
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]  # Close the plot

plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)

# Plot radar chart
ax.plot(angles, np.append(st_values, st_values[0]), 'o-', linewidth=1.5)
ax.fill(angles, np.append(st_values, st_values[0]), alpha=0.3)

# Axis settings
ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], labels)
ax.set_rlabel_position(0)
plt.yticks([0.0, 0.05, 0.10], ["0", "0.05", "0.10"], color="grey", size=8)
plt.ylim(0, 0.12)

# Add title
plt.title('Parameter Sensitivity Radar Chart (ST Index)', pad=20)
plt.tight_layout()
plt.savefig('figure2_parameter_radar.pdf')

# ====================== Figure 3: Normality Comparison ======================
# Data selection
peak_time = normality_test.loc[normality_test['p_value'].idxmax()]  # 9.4h
initial_time = normality_test.query('time == 0.1').iloc[0]  # Initial phase

# Generate simulated data
np.random.seed(42)
def generate_distribution(mean, std, n=1000):
    return np.random.normal(mean, std, n)

# Generate distributions
t1_data = generate_distribution(time_stability.loc[1, 'mean'], time_stability.loc[1, 'std'])
t2_data = generate_distribution(time_stability.loc[94, 'mean'], time_stability.loc[94, 'std'])

plt.figure(figsize=(8, 5))

# Plot KDE curves
sns.kdeplot(t1_data, color='#E74C3C', label=f'Initial Phase (0.1h)\nW={initial_time.W_stat:.2f}, p={initial_time.p_value:.1e}')
sns.kdeplot(t2_data, color='#3498DB', label=f'Stable Phase (9.4h)\nW={peak_time.W_stat:.3f}, p={peak_time.p_value:.3f}')

# Styling
plt.title('Normality Comparison of Wear Distribution', pad=15)
plt.xlabel('Wear Q')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure3_normality_compare.pdf')

plt.show()