import csv  # For reading and parsing CSV files
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For creating plots
from datetime import datetime, timedelta  # For handling date and time operations
from sklearn.linear_model import LinearRegression  # For performing linear regression

def load_csv(file_path):
    """
    Load stock data from a CSV file and return dates and prices.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        list: A list of datetime objects representing dates.
        list: A list of floats representing stock prices.
    """
    dates = []  # List to store dates
    prices = []  # List to store stock prices
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)  # Read the CSV file as a dictionary
        for row in reader:
            # Parse the date from the CSV and convert to datetime object
            dates.append(datetime.strptime(row['Date'], '%m/%d/%Y'))
            # Parse the price from the CSV and convert to float
            prices.append(float(row['High'].replace('$', '').replace(',', '')))
    # Ensure dates and prices are sorted in ascending order of dates
    sorted_data = sorted(zip(dates, prices))
    dates, prices = zip(*sorted_data)
    return list(dates), list(prices)

def predict_with_linear_regression(dates, prices, future_days=60, training_window=30):
    """
    Predict future stock prices using Linear Regression.

    Args:
        dates (list): List of dates for historical stock prices.
        prices (list): List of corresponding stock prices.
        future_days (int): Number of days to predict into the future.
        training_window (int): Number of most recent data points to use for training.

    Returns:
        np.ndarray: Future x-values (indices for predicted days).
        np.ndarray: Predicted stock prices for future days.
    """
    # Use only the last `training_window` data points for training
    x_train = np.arange(len(dates))[-training_window:].reshape(-1, 1)  # Indices of the training data
    y_train = np.array(prices)[-training_window:]  # Prices for the training data

    # Initialize and train the Linear Regression model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Predict stock prices for the next `future_days`
    x_future = np.arange(len(dates), len(dates) + future_days).reshape(-1, 1)  # Indices for future dates
    y_pred = model.predict(x_future)  # Predicted stock prices

    return x_future.flatten(), y_pred

def plot_linear_regression_prediction(dates, prices, future_days, predictions, training_window):
    """
    Plot historical stock data and Linear Regression predictions.

    Args:
        dates (list): List of historical dates.
        prices (list): List of historical stock prices.
        future_days (int): Number of days predicted into the future.
        predictions (tuple): Predicted x-values and y-values.
        training_window (int): Number of data points used for training.
    """
    x_future, predicted_prices = predictions  # Unpack future indices and predicted prices

    # Generate future dates starting from the last historical date
    last_date = dates[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]

    plt.figure(figsize=(12, 6))  # Set figure size for the plot

    # Plot the entire historical data
    plt.plot(dates, prices, label="Historical Data", color="green", marker="o", linewidth=1)

    # Highlight the training data used for the regression
    plt.plot(dates[-training_window:], prices[-training_window:], label="Training Data", color="orange", marker="o", linewidth=1)

    # Plot the predictions from Linear Regression
    plt.plot(future_dates, predicted_prices, label="Linear Regression Prediction", linestyle="--", color="blue")

    # Add labels, title, legend, and grid to the plot
    plt.xlabel("Date")
    plt.ylabel("Stock Price ($)")
    plt.title("Stock Price Analysis with Linear Regression Prediction")
    plt.legend()
    plt.grid()
    plt.show()

# Main Script
file_path = "META.csv"  # Path to the CSV file containing stock data
dates, prices = load_csv(file_path)  # Load dates and prices from the CSV

future_days = 60  # Number of days to predict into the future
training_window = 300  # Number of most recent data points to use for training

# Predict stock prices using Linear Regression
linear_predictions = predict_with_linear_regression(dates, prices, future_days, training_window)

# Plot the historical data and predictions
plot_linear_regression_prediction(dates, prices, future_days, linear_predictions, training_window)