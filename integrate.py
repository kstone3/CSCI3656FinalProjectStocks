import csv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import lagrange_spline as ls
import linear_regression as lr


def trapezoidal_single_interval(func, a, b):
    return 0.5 * (b - a) * (func(a) + func(b))

def legendre_polynomial(n, x):
    """Compute the value of the n-th Legendre polynomial at x."""
    if n == 0: return 1
    elif n == 1: return x
    else: return ((2 * n - 1) * x * legendre_polynomial(n - 1, x) - (n - 1) * legendre_polynomial(n - 2, x)) / n

def legendre_polynomial_derivative(n, x):
    """Compute the derivative of the n-th Legendre polynomial at x."""
    return n / (x ** 2 - 1) * (x * legendre_polynomial(n, x) - legendre_polynomial(n - 1, x))

def gaussian_quadrature_nodes_weights(n):
    """
    Compute nodes and weights for Gaussian Quadrature of order n.
    Returns nodes (roots of Legendre polynomial) and weights.
    """
    nodes = np.cos(np.pi * (np.arange(1, n + 1) - 0.25) / (n + 0.5))
    for _ in range(100): nodes -= legendre_polynomial(n, nodes) / legendre_polynomial_derivative(n, nodes)
    weights = 2 / ((1 - nodes ** 2) * (legendre_polynomial_derivative(n, nodes) ** 2))
    return nodes, weights

def gaussian_quadrature(func, a, b, n=3):
    """
    Gaussian Quadrature for a single interval [a, b].
    n: Number of points (accuracy level).
    """
    nodes,weights=gaussian_quadrature_nodes_weights(n)
    return 0.5 * (b - a) * np.sum(weights * func( 0.5 * (b - a) * nodes + 0.5 * (b + a)))

def integrate_stock_prices(func, indices, total_shares, method='trapezoidal'):
    daily_prices = []
    for i in indices[:-1]:
        if method == 'trapezoidal': daily_prices.append(trapezoidal_single_interval(func, i, i+1) * total_shares[i])
        elif method == 'gaussian': daily_prices.append(gaussian_quadrature(func, i, i+1) * total_shares[i])
    return daily_prices

def load_csv(file_path):
    dates = []
    total_shares = []
    prices=[]
    with open(file_path, 'r') as file:
        for row in csv.DictReader(file):
            dates.append(datetime.strptime(row['Date'], '%m/%d/%Y'))
            total_shares.append(float(row['Volume']))
            prices.append(float(row['High'].replace('$', '').replace(',', '')))
    return dates, prices,total_shares

def main():
    file_path = input("Enter the CSV file path: ")
    #file_path='SPY.csv'
    dates, prices,total_shares = load_csv(file_path)

    print("Select a stock price model:")
    print("1. Linear Regression")
    print("2. Lagrange Interpolation")
    print("3. Cubic Spline Interpolation")
    model_choice = int(input("Enter your choice (1/2/3): "))
    
    x_indices = np.arange(len(dates))
    
    if model_choice == 1:
        x_future, y_pred = lr.predict_with_linear_regression(dates, prices,training_window=2000)
        stock_price_func = lambda t: np.interp(t, x_future, y_pred)
    elif model_choice == 2: 
        indices=np.linspace(0, len(dates) - 1, num=12, dtype=int)
        stock_price_func = lambda t: (ls.lagrange_interpolation(indices, (np.array(prices)[indices]), [t])[0])
    elif model_choice == 3: stock_price_func = lambda t: (ls.cubic_spline_interpolation(x_indices, prices, [t])[0])
    else:
        print("Invalid choice. Exiting.")
        return

    print("Select an integration method:")
    print("1. Trapezoidal Rule")
    print("2. Gaussian Quadrature")
    integration_choice = int(input("Enter your choice (1/2): "))
    integration_method = 'trapezoidal' if integration_choice == 1 else 'gaussian'

    # Debug: Verify stock price function behavior
    # print("Testing stock_price_func at sample points:")
    # for t in np.linspace(0, len(dates) - 1, 5):  # Test over dataset index range
    #     t=int(round(t))
    #     try:
    #         print(f"t={t:.2f}, stock_price_func={stock_price_func(t):.2f}")
    #     except Exception as e:
    #         print(f"t={t:.2f}, Error: {e}")

    # Integrate stock prices
    integrated_prices = integrate_stock_prices(stock_price_func, x_indices, total_shares,method=integration_method)
    # Debug output
    print("Integration complete. Plotting results...")

    plt.figure(figsize=(12, 6))
    plt.plot(dates[:-1], integrated_prices, label="Integrated Stock Price")
    plt.xlabel("Date")
    plt.ylabel("Stock Price ($)")
    plt.title("Stock Price Integration")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
