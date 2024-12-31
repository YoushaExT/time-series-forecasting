import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)


def evaluate_model(test, forecast):
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)
    r2 = r2_score(test, forecast)
    mape = mean_absolute_percentage_error(test, forecast)
    return rmse, mae, r2, mape


def plot(data, cols):
    # plot the time series data
    plt.figure(figsize=(15, 6))
    plt.plot(data[cols[0]], data[cols[1]])
    plt.title("TSF")
    plt.xlabel(cols[0])
    plt.ylabel(cols[1])
    plt.show()


def plot_model_results(
    train, test, forecast, forecast_train, label, cols, differencing=0
):
    _train = train.copy()
    _test = test.copy()
    if differencing:
        _train = _train.iloc[differencing:]
        _test = _test.iloc[differencing:]

    # Plot Exponential Smoothing results
    plt.figure(figsize=(15, 6))
    plt.plot(train[cols[0]], train[cols[1]], label="Train", color="blue")
    plt.plot(test[cols[0]], test[cols[1]], label="Test", color="green")
    plt.plot(_train[cols[0]], forecast_train, label="In-sample Forecast", color="red")
    plt.plot(
        _test[cols[0]],
        forecast,
        label=label,
        color="orange",
    )
    plt.legend(loc="best")
    plt.show()
