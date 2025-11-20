import numpy as np

# Define matrix A
A = np.array([[4, 1],
              [2, 3]])

print("Matrix A:\n", A)

# Compute eigenvalues & eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("\nEigenvalues:")
print(eigenvalues)

print("\nEigenvectors (columns represent vectors):")
print(eigenvectors)

# Diagonalization
D = np.diag(eigenvalues)        # diagonal matrix
P = eigenvectors                # eigenvector matrix
P_inv = np.linalg.inv(P)        # inverse of P

A_reconstructed = P @ D @ P_inv

print("\nDiagonal Matrix D:\n", D)
print("\nP * D * P^-1 (Should give A):\n", A_reconstructed)
