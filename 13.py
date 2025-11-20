import numpy as np

# ----------------- Divided Difference Table -----------------
def divided_diff(x, y):
    n = len(x)
    table = np.zeros((n, n))
    table[:, 0] = y  # first column = y values

    for j in range(1, n):             # for each column
        for i in range(n - j):        # for each row
            table[i][j] = (table[i+1][j-1] - table[i][j-1]) / (x[i+j] - x[i])
    return table

# ----------------- Newton Interpolation Polynomial -----------------
def newton_interpolation(x, table, value):
    n = len(x)
    result = table[0, 0]      # a0
    product = 1               # (x - x0), (x - x0)(x - x1)...

    for i in range(1, n):
        product *= (value - x[i-1])
        result += table[0][i] * product
    return result

# ----------------- Example -----------------
x_points = np.array([1, 2, 3, 4])
y_points = np.array([1, 4, 9, 16])  # f(x) = x^2

table = divided_diff(x_points, y_points)
value = 2.5
print("Interpolated Value:", newton_interpolation(x_points, table, value))
