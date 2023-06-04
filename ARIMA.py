import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 读取csv数据
data = pd.read_csv('Wordle2.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

# 数据可视化
plt.figure(figsize=(12, 6))
plt.plot(data)
plt.title('Time Series')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

# ACF和PACF分析
plt.figure(figsize=(12, 6))
plot_acf(data.Value, lags=30)  # 绘制ACF图
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.show()

plt.figure(figsize=(12, 6))
plot_pacf(data.Value, lags=30)  # 绘制PACF图
plt.xlabel('Lags')
plt.ylabel('PACF')
plt.show()



# 拟合ARIMA模型
model = ARIMA(data.Value, order=(1, 1, 0))
# model = VAR(data)
model_fit = model.fit()


# 输出模型的AIC评分和参数
print('AIC Score: %.2f' % model_fit.aic)
print('Coefficients: %s' % model_fit.params)
# print(model_fit)

n_days = 7
forecast = model_fit.forecast(steps=n_days)

# 输出预测结果
# print('Forecast Result:')
# print(forecast)

# plt.plot(data, label='Original Data')
plt.plot(forecast, label='Predicted Data')
# plt.legend()
plt.show()
