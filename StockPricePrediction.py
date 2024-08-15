# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load stock data using yfinance
ticker = 'AAPL'  # Example: Apple Inc.
stock_data = yf.download(ticker, start="2010-01-01", end="2023-01-01")

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(stock_data.head())

# Visualize the stock's closing price history
plt.figure(figsize=(14, 7))
plt.plot(stock_data['Close'], label='Close Price History')
plt.title(f'{ticker} Closing Price History')
plt.xlabel('Date')
plt.ylabel('Closing Price USD ($)')
plt.legend()
plt.show()

# Feature Engineering: Use 'Close' prices as the target variable
stock_data['Prediction'] = stock_data['Close'].shift(-30)  # Predicting 30 days into the future

# Create the feature dataset (X) and the target dataset (y)
X = np.array(stock_data[['Close']])[:-30]
y = np.array(stock_data['Prediction'])[:-30]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Visualize the results
plt.figure(figsize=(14, 7))
plt.plot(y_test, color='blue', label='Actual Prices')
plt.plot(y_pred, color='red', label='Predicted Prices')
plt.title(f'{ticker} Actual vs Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Price USD ($)')
plt.legend()
plt.show()

# Predicting the next 30 days
X_future = stock_data[['Close']][-30:]
X_future_scaled = scaler.transform(X_future)
future_prediction = model.predict(X_future_scaled)

# Visualizing the future prediction
plt.figure(figsize=(14, 7))
plt.plot(stock_data['Close'], label='Closing Price History')
plt.plot(pd.date_range(start=stock_data.index[-1], periods=30, freq='B'), future_prediction, color='orange', label='Future Predictions')
plt.title(f'{ticker} Closing Price Prediction for the Next 30 Days')
plt.xlabel('Date')
plt.ylabel('Price USD ($)')
plt.legend()
plt.show()

# Optional: Save the model
import joblib
joblib.dump(model, 'stock_price_predictor.pkl')
print("\nModel saved as 'stock_price_predictor.pkl'")
