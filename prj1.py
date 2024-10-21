import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
# Fetch historical data for a stock (e.g., Apple Inc.)
stock_data = yf.download('AAPL', start='2010-01-01', end='2024-01-01')
print(stock_data.head())


# Create a new DataFrame with only the 'Close' price
data = stock_data[['Close']].copy()

# Create features and labels
data['Return'] = data['Close'].pct_change()  # Daily returns
data['Lag1'] = data['Close'].shift(1)        # Previous day's closing price
data.dropna(inplace=True)                    # Remove NaN values

# Define features (X) and target (y)
X = data[['Lag1']]
y = data['Return']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)


mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae}')


plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Returns')
plt.scatter(X_test, predictions, color='red', label='Predicted Returns')
plt.xlabel('Previous Day Closing Price')
plt.ylabel('Daily Return')
plt.title('Actual vs Predicted Returns')
plt.legend()
plt.show()
