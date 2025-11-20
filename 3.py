import numpy as np

A = np.array([[1, 1, 1],
              [2, 3, 1],
              [1, 2, 3]], dtype=float)

B = np.array([6, 14, 14], dtype=float)
n = len(B)

# ---------- Forward Elimination ----------
for i in range(n):
    for k in range(i+1, n):
        factor = A[k][i] / A[i][i]
        A[k] = A[k] - factor * A[i]
        B[k] = B[k] - factor * B[i]

print("Upper Triangular Matrix A:\n", A)
print("Modified B:\n", B)

# ---------- Backward Substitution ----------
x = [0] * n  # Initialize solution vector
for i in range(n-1, -1, -1):  # Start from last row, go upwards
    ax = 0
    for j in range(i+1, n):
        ax += A[i][j] * x[j]  # Sum known variables
    x[i] = (B[i] - ax) / A[i][i]  # Solve for current variable

print("\nSolution Vector [x, y, z]:")
print(x)
