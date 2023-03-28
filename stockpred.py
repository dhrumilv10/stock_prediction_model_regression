import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random
from alpha_vantage.timeseries import TimeSeries
from sklearn.linear_model import LinearRegression

# Define AlphaVantage API key
API_LIST = ['C62KJYK2FP4UPYS5', 'OHJT76WFI9Q5KYWC', 'SAMG23F9CPTWYJMH', 'X7IJVSVGCUKO8JVL', 'D9IVGB2JF0WTM2HI', 'R4EZB4WRPT1J99SM', 'E07BCT34EGXBHS92', 'LRT8HMGIZ1V6O872', 'ZYBW873BN926G3V0', 'IB3A28NTZWM73RY1', 'B5FSD9WMZ6U4NYM7', '49K6TT296T8UCY4O', '7MXW85ZU4P75R9S2', '54RQXMZN07QZT7Q3', 'QQI0E421R46VODUK']
ALPHAVANTAGE_API_KEY = random.choice(API_LIST)

# Define time period for simple moving average and exponential moving average
TIME_PERIOD = 20


# Define function to get stock data from AlphaVantage
def get_stock_data(symbol, interval):
    ts = TimeSeries(key=ALPHAVANTAGE_API_KEY, output_format='pandas')
    data, meta_data = ts.get_intraday(symbol=symbol, interval=interval, outputsize='full')
    data = data.iloc[::-1]
    data.index = pd.to_datetime(data.index)
    close_price = data['4. close']
    open_price = data['1. open']
    high_price = data['2. high']
    low_price = data['3. low']
    volume = data['5. volume']
    sma = close_price.rolling(window=TIME_PERIOD).mean()
    ema = close_price.ewm(span=TIME_PERIOD, adjust=False).mean()
    rsi = 100 - (100 / (1 + (
            close_price.diff().fillna(0).rolling(TIME_PERIOD).mean().abs() / close_price.diff().fillna(0).rolling(
        TIME_PERIOD).mean()).fillna(0)))
    macd = ema.ewm(span=12, adjust=False).mean() - ema.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    data = pd.concat([close_price, open_price, high_price, low_price, volume, sma, ema, rsi, macd, signal, histogram],
                     axis=1)
    data.columns = ['Close', 'Open', 'High', 'Low', 'Volume', 'SMA', 'EMA', 'RSI', 'MACD', 'Signal', 'Histogram']
    return data


# Define function to plot stock data
def plot_stock_data(data):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(data.index, data['Close'], label='Close Price')
    ax.legend()
    ax.spines['bottom'].set_visible(False)  # hide the x-axis spine
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # hide the ticks and labels
    # for the x-axis
    st.pyplot(fig)


# Define function to make stock price predictions
def predict_stock(data, days):
    features = ['Close', 'Open', 'High', 'Low', 'Volume', 'SMA', 'EMA', 'RSI', 'MACD', 'Signal', 'Histogram']
    X = data[features].values
    y = data['Close'].values
    n_train = len(data) - days
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    future_index = pd.date_range(start=data.index[-1], periods=days + 1, freq='B')[1:]
    future_data = pd.DataFrame(index=future_index, columns=data.columns)
    future_data.iloc[0] = data.iloc[-1].values
    for i in range(1, len(future_data)):
        X_future = future_data.iloc[i - 1][features].values.reshape(1, -1)
        future_data.iloc[i]['Close'] = model.predict(X_future)
        future_data.iloc[i]['Open'] = future_data.iloc[i - 1]['Close']
        future_data.iloc[i]['High'] = future_data.iloc[i]['Close'] * 1.02
        future_data.iloc[i]['Low'] = future_data.iloc[i]['Close'] * 0.98
        future_data.iloc[i]['Volume'] = 100000
        future_data.iloc[i]['SMA'] = future_data.iloc[max(i - TIME_PERIOD, 0):i]['Close'].mean()
        future_data.iloc[i]['EMA'] = \
            future_data.iloc[max(i - TIME_PERIOD, 0):i]['Close'].ewm(span=TIME_PERIOD, adjust=False).mean().iloc[-1]
        future_data.iloc[i]['RSI'] = 100 - (100 / (1 + (
                future_data.iloc[max(i - TIME_PERIOD, 0):i]['Close'].diff().fillna(0).rolling(
                    TIME_PERIOD).mean().abs() / future_data.iloc[max(i - TIME_PERIOD, 0):i]['Close'].diff().fillna(
            0).rolling(TIME_PERIOD).mean()).fillna(0))).iloc[-1]
        future_data.iloc[i]['MACD'] = \
            future_data.iloc[max(i - TIME_PERIOD, 0):i]['Close'].ewm(span=12, adjust=False).mean().iloc[-1] - \
            future_data.iloc[max(i - TIME_PERIOD, 0):i]['Close'].ewm(span=26, adjust=False).mean().iloc[-1]
        future_data.iloc[i]['Signal'] = \
            future_data.iloc[max(i - TIME_PERIOD, 0):i]['MACD'].ewm(span=9, adjust=False).mean().iloc[-1]
        future_data.iloc[i]['Histogram'] = future_data.iloc[i]['MACD'] - future_data.iloc[i]['Signal']
    return future_data


# Define main function
def main():
    # Add page title
    st.title('Stock Price Prediction')

    # Add user input section
    st.sidebar.header('User Input')
    symbol = st.sidebar.text_input('Enter stock symbol', 'AAPL')
    interval = st.sidebar.selectbox('Select time interval', ('1min', '5min', '15min', '30min', '60min'))
    days = st.sidebar.slider('Select number of days to predict', 1, 30, 7)

    # Get stock data
    data = get_stock_data(symbol, interval)

    # Plot stock data
    plot_stock_data(data)

    # Make stock price predictions
    predict_stock(data, days)


if __name__ == '__main__':
    main()