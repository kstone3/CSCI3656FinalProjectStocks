import csv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime,timedelta

coefficients = [2, -3, 4, 5]

def stock_price(t):
    value = 0
    for i, coeff in enumerate(reversed(coefficients)): value += coeff * (t ** i)
    return value

#Can change to use more advanced rule
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
    # Initial guesses for roots using Chebyshev nodes
    nodes = np.cos(np.pi * (np.arange(1, n + 1) - 0.25) / (n + 0.5))

    # Refine roots using Newton's method
    for _ in range(100):  # Iterate to improve precision
        nodes -= legendre_polynomial(n, nodes) / legendre_polynomial_derivative(n, nodes)

    # Compute weights
    weights = 2 / ((1 - nodes ** 2) * (legendre_polynomial_derivative(n, nodes) ** 2))

    return nodes, weights

def gaussian_quadrature(func, a, b, n=3):
    """
    Gaussian Quadrature for a single interval [a, b].
    n: Number of points (accuracy level).
    """
    # Nodes and weights for n=3 (Legendre-Gauss for [-1, 1])
    # Can extend for higher n if needed
    nodes,weights=gaussian_quadrature_nodes_weights(n)
    return 0.5 * (b - a) * np.sum(weights * func( 0.5 * (b - a) * nodes + 0.5 * (b + a)))


def compute_daily_stock_prices(func, total_shares, start_date, end_date):
    current_date = start_date
    daily_prices = []
    dates = []
    i=len(total_shares)-1
    while current_date < end_date:
        t_start = (current_date - start_date).days / 365.0
        t_end = t_start + 1 / 365.0
        daily_prices.append(gaussian_quadrature(func, t_start, t_end) * total_shares[i])
        #daily_prices.append(trapezoidal_single_interval(func, t_start, t_end)*total_shares[i])
        dates.append(current_date)
        i-=1
        current_date += timedelta(days=1)
    return dates, daily_prices

def load_csv(file_path):
    dates = []
    total_shares = []
    with open(file_path, 'r') as file:
        for row in csv.DictReader(file):
            dates.append(datetime.strptime(row['Date'], '%m/%d/%Y'))
            total_shares.append(float(row['Volume']))
    return dates, total_shares

dates, total_shares = load_csv("META.csv")
dates, daily_stock_prices = compute_daily_stock_prices(stock_price, total_shares,dates[-1], dates[0])
plt.figure(figsize=(12, 6))
plt.plot(dates, daily_stock_prices, label="Daily Stock Price")
plt.xlabel("Date")
plt.ylabel("Stock Price ($)")
plt.title("Stock Prices Per Day (2014-2024) Using Trapezoidal Rule")
plt.legend()
plt.grid()
plt.show()
