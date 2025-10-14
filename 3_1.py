# Assignment 3: Solve System of Linear Equations using NumPy

import numpy as np

# Step 1: Define coefficient matrix (A) and constant matrix (B)
A = np.array([[1, 1, 1],
              [2, 3, 1],
              [1, 2, 3]], dtype=float)

B = np.array([6, 14, 14], dtype=float)

print("Coefficient Matrix (A):")
print(A)
print("\nConstant Vector (B):")
print(B)

# Step 2: Solve the system AX = B
X = np.linalg.solve(A, B)

# Step 3: Display the solution
print("\nSolution Vector [x, y, z]:")
print(X)

# Step 4: Verify the result (A·X should equal B)
verification = np.allclose(np.dot(A, X), B)
print("\nVerification (A·X = B):", verification)
