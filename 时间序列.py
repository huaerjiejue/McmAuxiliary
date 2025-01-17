import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
import pmdarima as pm

# 示例数据：生成随机时间序列
np.random.seed(0)
data = np.random.randn(100).cumsum()

# 绘制 ACF 和 PACF 图
plot_acf(data, lags=20)
plot_pacf(data, lags=20)
plt.show()

# 使用 STL 分解时间序列
stl = STL(data, period=12)
res = stl.fit()
fig = res.plot()
plt.show()

model = ARIMA(data, order=(1, 1, 1))
model_fit = model.fit()
print(model_fit.summary())
# fc, conf = model_fit.forecast(25, alpha=0.05)

# 使用 pmdarima 库自动选择 ARIMA 模型
model = pm.auto_arima(data, start_p=1, start_q=1,
                      test='adf',
                      max_p=3, max_q=3, m=12,
                      start_P=0, seasonal=True,
                      d=None, D=1, trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)
print(model.summary())
fc, conf = model.predict(n_periods=25, return_conf_int=True)
print(fc)
print(conf)

# plot
plt.plot(data, label='observed')
plt.plot(np.arange(100, 125), fc, label='forecast')
plt.fill_between(np.arange(100, 125), conf[:, 0], conf[:, 1], color='k', alpha=0.1)
plt.legend()
plt.show()