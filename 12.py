import numpy as np
import matplotlib.pyplot as plt

# Given points
x_points = np.array([1, 2, 3])
y_points = np.array([2, 3, 5])

# Lagrange Interpolation Function
def lagrange_interpolate(x_points, y_points, x):
    total = 0
    n = len(x_points)
    
    for i in range(n):
        xi, yi = x_points[i], y_points[i]
        term = yi
        
        for j in range(n):
            if i != j:
                xj = x_points[j]
                term *= (x - xj) / (xi - xj)
        
        total += term
    return total

# Generate x-values for smooth curve
x_new = np.linspace(min(x_points), max(x_points), 200)
y_new = lagrange_interpolate(x_points, y_points, x_new)

# Plot
plt.scatter(x_points, y_points, color='red', label="Given Points")
plt.plot(x_new, y_new, color='blue', label="Lagrange Polynomial")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Lagrange Interpolation")
plt.legend()
plt.grid(True)
plt.show()
