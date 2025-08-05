import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

url = "BrentOilPrices.csv"

df = pd.read_csv(url)

df.head()

# Data Cleaning

# Check for Missing Data
df.isnull().sum()
# Remove rows with missing values
df.dropna()
# Convert datetime
df['Date'] = pd.to_datetime(df['Date'])
# Calculate the mean for 30 days
df['Price'] = df['Price'].rolling(window=30).mean()

# Look for trends
df.plot(title='Time Series Data')

# Test whether it's Time Series Stationary or not?
# Test ADF with 'Price'
result = adfuller(df['Price'].dropna())

print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

# Do Differencing
df['Price_diff'] = df['Price'].diff()
df_stationary = df['Price_diff'].dropna()

# Test ADF with differencing data
result_differencing = adfuller(df_stationary)
print(f'ADF Statistic (diff): {result_differencing[0]}')
print(f'p-value (diff): {result_differencing[1]}')

# Use ACF and PACF plot to find (p, d, q)
plot_acf(df_stationary, lags=30)
plot_pacf(df_stationary, lags=30)
plt.show()

# ARIMA model
p = 2
d = 1
q = 1

model = ARIMA(df_stationary.dropna(), order=(p, d, q))
model_fit = model.fit()

print(model_fit.summary())

# Forecasting
forecast = model_fit.forecast(steps=12)
forecast.plot()
plt.title("Forecast")
plt.show()
