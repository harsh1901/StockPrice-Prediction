#pip install -r requirement.txt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import yfinance as yf

# Ask user for stock ticker
ticker = input("Enter stock ticker (e.g., AAPL, TSLA, MSFT, INFY): ").upper()

# Download stock data
data = yf.download(ticker, start='2018-01-01', end='2026-01-01')

# Use only the 'Close' price
data = data[['Close']]
data['Prediction'] = data[['Close']].shift(-30)  # Predict next 30 days

# Features (X) and Labels (y)
X = np.array(data.drop(['Prediction'], axis=1))[:-30]
y = np.array(data['Prediction'])[:-30]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Root Mean Squared Error for {ticker}: {rmse}")

# Predict next 30 days
x_future = np.array(data.drop(['Prediction'], axis=1))[-30:]
future_predictions = model.predict(x_future)

# Plot results
valid = data[-30:].copy()
valid['Predictions'] = future_predictions

plt.figure(figsize=(12, 6))
plt.title(f'Stock Price Prediction ({ticker})')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.plot(data['Close'], label='Historical')
plt.plot(valid[['Predictions']], label='Predicted', color='YELLOW')
plt.legend()

plt.show()
