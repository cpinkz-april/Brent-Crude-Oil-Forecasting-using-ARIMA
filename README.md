# ⛽ Brent Crude Oil Price Forecasting using ARIMA

This project applies the **ARIMA (AutoRegressive Integrated Moving Average)** model to forecast Brent Crude Oil Prices.<br>
The goal is to build a time series model that can predict short-term future prices.

## 📊 Dataset 

- **Source**: Brent Crude Oil Prices
- **File**: `BrentOilPrices.csv`
- **Columns**:
  - `Date`: Date
  - `Price`: Brent Crude Oil Price (USD/barrel)

## 🎯 Objectives

- Look for oil price trends
- Test for Augmented Dickey-Fuller (ADF)
- Apply differencing to make the series stationary (if required)
- Use ACF/PACF plots to determine ARIMA(p,d,q)
- Train the ARIMA model
- Forecast oil prices for the next 12 months
- Visualize forecast results

## 🧰 Libraries Used

- Python 
- pandas
- matplotlib
- statsmodels
- scikit-learn
