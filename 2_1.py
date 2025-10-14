# Assignment 2: Eigenvalues, Eigenvectors, and Diagonalization using NumPy

import numpy as np

# Step 1: Define a square matrix
A = np.array([[4, 1],
              [2, 3]])

print("Matrix A:")
print(A)

# Step 2: Compute Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("\nEigenvalues:")
print(eigenvalues)

print("\nEigenvectors (columns correspond to eigenvalues):")
print(eigenvectors)

# Step 3: Diagonalization
# P = matrix of eigenvectors
# D = diagonal matrix of eigenvalues
P = eigenvectors
D = np.diag(eigenvalues)

# Step 4: Verify A = P * D * P⁻¹
A_reconstructed = P @ D @ np.linalg.inv(P)

print("\nDiagonal Matrix D:")
print(D)

print("\nReconstructed Matrix (P * D * P⁻¹):")
print(A_reconstructed)

# Step 5: Check if reconstruction is correct
if np.allclose(A, A_reconstructed):
    print("\n✅ Matrix A successfully diagonalized and reconstructed.")
else:
    print("\n❌ Diagonalization failed.")
