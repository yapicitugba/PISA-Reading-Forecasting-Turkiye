"""
Training templates for GRU, LSTM, ARIMA, and Prophet models.

Note:
Because the article reports final metric and forecast values, this file is provided as a reproducible
template for how the models can be trained. Exact numerical reproduction may depend on random seed,
software versions, and package availability.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

def create_sliding_windows(values, window_size=3):
    X, y = [], []
    values = np.asarray(values, dtype=float)
    for i in range(len(values) - window_size):
        X.append(values[i:i + window_size])
        y.append(values[i + window_size])
    return np.asarray(X), np.asarray(y)

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

def recursive_forecast(model_predict_function, initial_window, steps=3):
    window = list(initial_window)
    forecasts = []
    for _ in range(steps):
        pred = float(model_predict_function(np.asarray(window).reshape(1, -1, 1)))
        forecasts.append(pred)
        window = window[1:] + [pred]
    return forecasts

def train_arima(series, order=(1, 1, 1), steps=3):
    model = ARIMA(series, order=order)
    fitted = model.fit()
    return fitted.forecast(steps=steps)

def train_gru_template(series, window_size=3, epochs=100, batch_size=4):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dense
    from tensorflow.keras.optimizers import Adam

    scaler = StandardScaler()
    scaled = scaler.fit_transform(np.asarray(series).reshape(-1, 1)).flatten()

    X, y = create_sliding_windows(scaled, window_size=window_size)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        GRU(50, activation="tanh", input_shape=(window_size, 1)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    return model, scaler

def train_lstm_template(series, window_size=3, epochs=100, batch_size=4):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.optimizers import Adam

    scaler = StandardScaler()
    scaled = scaler.fit_transform(np.asarray(series).reshape(-1, 1)).flatten()

    X, y = create_sliding_windows(scaled, window_size=window_size)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, activation="tanh", input_shape=(window_size, 1)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    return model, scaler

def train_prophet_template(df):
    # df must contain columns: ds, y
    from prophet import Prophet
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=3, freq="3Y")
    forecast = model.predict(future)
    return model, forecast

if __name__ == "__main__":
    turkiye_reading = [441, 447, 464, 475, 428, 466, 456]
    print("ARIMA forecast example:", train_arima(turkiye_reading, order=(1, 1, 1), steps=3))
