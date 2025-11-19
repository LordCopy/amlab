import numpy as np
import matplotlib.pyplot as plt

# Practical 12: Implement Lagrange's Interpolation Method

def lagrange_interpolation(x_points, y_points, x_val):
    n = len(x_points)
    result = 0.0
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if i != j:
                term *= (x_val - x_points[j]) / (x_points[i] - x_points[j])
        result += term
    return result

if __name__ == '__main__':
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 1, 3, 7])

    x_new = 2.5
    y_new = lagrange_interpolation(x, y, x_new)
    print(f"Interpolated value at x = {x_new} is y = {y_new}")

    x_range = np.linspace(min(x), max(x), 100)
    y_range = [lagrange_interpolation(x, y, xi) for xi in x_range]

    plt.scatter(x, y, color='red', label='Data Points')
    plt.plot(x_range, y_range, color='blue', label='Lagrange Interpolation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Lagrange Interpolation")
    plt.legend()
    plt.grid(True)
    plt.show()
