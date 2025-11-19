import numpy as np
import matplotlib.pyplot as plt

# Practical 12 — Lagrange interpolation

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
    x = np.array([1,2,3,4,5])
    y = np.array([2,4,1,3,7])
    x_new = 2.5
    print('Interpolated:', lagrange_interpolation(x,y,x_new))
    xr = np.linspace(min(x), max(x), 100)
    yr = [lagrange_interpolation(x,y,xi) for xi in xr]
    plt.scatter(x,y); plt.plot(xr,yr); plt.show()
