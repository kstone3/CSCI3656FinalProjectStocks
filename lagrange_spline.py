import csv
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For visualization
from datetime import datetime  # To handle date formatting

def load_csv(file_path):
    """
    Load stock data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        list: A list of datetime objects representing stock dates.
        list: A list of floats representing the corresponding stock prices.
    """
    dates = []  # To store parsed dates
    prices = []  # To store the corresponding stock prices
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Convert date from string format to datetime object
            dates.append(datetime.strptime(row['Date'], '%m/%d/%Y'))
            # Convert price from string to float, removing '$' and ',' for processing
            prices.append(float(row['High'].replace('$', '').replace(',', '')))
    return dates, prices

def lagrange_interpolation(x_subset, y_subset, x_eval):
    """
    Manually perform Lagrange interpolation.

    Args:
        x_subset (list or np.ndarray): Known x-values (indices).
        y_subset (list or np.ndarray): Corresponding y-values (prices) for x_subset.
        x_eval (list or np.ndarray): x-values where interpolation should be evaluated.

    Returns:
        np.ndarray: Interpolated y-values for x_eval.
    """
    n = len(x_subset)  # Number of known points
    interpolated = []  # List to store interpolated values
    for x in x_eval:
        P_x = 0  # Initialize the Lagrange polynomial for the current x
        for i in range(n):
            L_i = 1  # Initialize Lagrange basis polynomial L_i(x)
            for j in range(n):
                if i != j:
                    # Compute the product of (x - x_j) / (x_i - x_j)
                    L_i *= (x - x_subset[j]) / (x_subset[i] - x_subset[j])
            # Add the contribution of the i-th term to the polynomial
            P_x += y_subset[i] * L_i
        interpolated.append(P_x)  # Append the interpolated value
    return np.array(interpolated)

def cubic_spline_interpolation(x, y, x_eval):
    """
    Manually perform cubic spline interpolation.

    Args:
        x (list or np.ndarray): Known x-values (indices).
        y (list or np.ndarray): Corresponding y-values (prices).
        x_eval (list or np.ndarray): x-values where interpolation should be evaluated.

    Returns:
        np.ndarray: Interpolated y-values for x_eval.
    """
    n = len(x) - 1  # Number of intervals between points
    h = np.diff(x)  # Compute the step size (x_{i+1} - x_i) for each interval
    b = np.diff(y) / h  # Compute the slope (y_{i+1} - y_i) / h

    # Set up the tridiagonal system for solving spline coefficients
    A = np.zeros((n+1, n+1))  # Coefficient matrix
    rhs = np.zeros(n+1)  # Right-hand side vector

    # Natural spline boundary conditions: second derivative at endpoints is zero
    A[0, 0] = 1
    A[n, n] = 1

    # Fill the tridiagonal system for continuity and smoothness
    for i in range(1, n):
        A[i, i-1] = h[i-1]  # Subdiagonal
        A[i, i] = 2 * (h[i-1] + h[i])  # Main diagonal
        A[i, i+1] = h[i]  # Superdiagonal
        # Right-hand side for the cubic spline conditions
        rhs[i] = 3 * (b[i] - b[i-1])

    # Solve the tridiagonal system for c coefficients (second derivatives)
    c = np.linalg.solve(A, rhs)

    # Compute b and d coefficients from the second derivatives
    b_coeff = b - h * (2 * c[:-1] + c[1:]) / 3  # Linear terms
    d_coeff = (c[1:] - c[:-1]) / (3 * h)  # Cubic terms

    # Compute a coefficients (the y-values at known points)
    a_coeff = y[:-1]
    if(isinstance(x_eval[0], np.ndarray)):
        x_eval=x_eval[0]
    # Interpolate the values at x_eval
    interpolated = []
    for x_val in x_eval:
        for i in range(n):
            if x[i] <= x_val <= x[i+1]:  # Check which interval x_val belongs to
                dx = x_val - x[i]  # Difference from the starting point of the interval
                # Compute the cubic spline value using the formula
                value = (a_coeff[i]+ b_coeff[i] * dx+ c[i] * dx**2+ d_coeff[i] * dx**3)
                interpolated.append(value)
                break  # Exit the loop once the interval is found
    return np.array(interpolated)

def interpolate_with_lagrange(dates, prices):
    """
    Interpolate stock prices using Lagrange interpolation.

    Args:
        dates (list): List of datetime objects representing stock dates.
        prices (list): List of floats representing stock prices.

    Returns:
        np.ndarray: Indices for interpolated values.
        np.ndarray: Interpolated stock prices.
    """
    indices = np.linspace(0, len(dates) - 1, num=12, dtype=int)  # Sample indices
    x_eval=np.arange(len(dates))
    return x_eval, lagrange_interpolation(indices, np.array(prices)[indices], x_eval)

def interpolate_with_splines(dates, prices):
    """
    Interpolate stock prices using cubic spline interpolation.

    Args:
        dates (list): List of datetime objects representing stock dates.
        prices (list): List of floats representing stock prices.

    Returns:
        np.ndarray: Indices for interpolated values.
        np.ndarray: Interpolated stock prices.
    """
    x = np.arange(len(dates))  # Treat dates as sequential indices for simplicity
    return x, cubic_spline_interpolation(x, prices, x)

def plot_stock_prices(dates, prices, lagrange_x, lagrange_prices, spline_x, spline_prices):
    """
    Plot original and interpolated stock prices.

    Args:
        dates (list): List of datetime objects representing stock dates.
        prices (list): List of floats representing original stock prices.
        lagrange_x (np.ndarray): Indices for Lagrange interpolation.
        lagrange_prices (np.ndarray): Interpolated stock prices using Lagrange interpolation.
        spline_x (np.ndarray): Indices for cubic spline interpolation.
        spline_prices (np.ndarray): Interpolated stock prices using cubic spline interpolation.
    """
    plt.figure(figsize=(12, 6))

    # Original data
    plt.plot(dates, prices, label="Original Data", marker="o", linewidth=0.5)

    # Lagrange interpolation
    plt.plot(dates, lagrange_prices, label="Lagrange Interpolation", linestyle="--", color="orange")

    # Spline interpolation
    plt.plot(dates, spline_prices, label="Spline Interpolation", linestyle="--", color="green")

    plt.xlabel("Date")
    plt.ylabel("Stock Price ($)")
    plt.title("Stock Price Analysis with Interpolation")
    plt.legend()
    plt.grid()
    plt.show()

# Load the stock price data from the CSV file
file_path = "META.csv"  # Update with your CSV file path
dates, prices = load_csv(file_path)

# Interpolate the current prices using Lagrange and cubic splines
lagrange_x, lagrange_prices = interpolate_with_lagrange(dates, prices)
spline_x, spline_prices = interpolate_with_splines(dates, prices)

# Plot the results
plot_stock_prices(dates, prices, lagrange_x, lagrange_prices, spline_x, spline_prices)