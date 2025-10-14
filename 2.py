import numpy as np

# Step 1: Define 2x2 matrix
A = np.array([[4, 1],
              [2, 3]])

print("Matrix A:")
print(A)

# Step 2: Compute eigenvalues manually
# For 2x2 matrix [[a, b], [c, d]]:
# Characteristic equation: |A - λI| = 0 → λ² - (a+d)λ + (ad - bc) = 0

a, b = A[0, 0], A[0, 1]
c, d = A[1, 0], A[1, 1]

trace = a + d
det = a * d - b * c

# Solve quadratic equation for eigenvalues
lambda1 = (trace + np.sqrt(trace**2 - 4 * det)) / 2
lambda2 = (trace - np.sqrt(trace**2 - 4 * det)) / 2

print("\nEigenvalues:")
print(lambda1, lambda2)

# Step 3: Compute eigenvectors manually
# Solve (A - λI)v = 0
# For 2x2: (a - λ)v1 + b*v2 = 0 → v2 = -((a - λ)/b) * v1

def eigenvector(A, lam):
    a, b = A[0, 0], A[0, 1]
    c, d = A[1, 0], A[1, 1]

    if b != 0:
        v1 = 1
        v2 = -((a - lam) / b) * v1
    elif c != 0:
        v2 = 1
        v1 = -((d - lam) / c) * v2
    else:  # diagonal matrix
        v1, v2 = 1, 0

    vec = np.array([v1, v2])
    return vec / np.linalg.norm(vec)  # normalize the vector

v1 = eigenvector(A, lambda1)
v2 = eigenvector(A, lambda2)

print("\nEigenvectors (normalized):")
print("v1 =", v1)
print("v2 =", v2)

# Step 4: Diagonalization
P = np.column_stack((v1, v2))  # matrix of eigenvectors
D = np.diag([lambda1, lambda2])  # diagonal matrix of eigenvalues
P_inv = np.linalg.inv(P)

A_diag = P @ D @ P_inv

print("\nDiagonal matrix D:")
print(D)
