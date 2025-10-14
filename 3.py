# Step 1: Define the system of equations
# A is the coefficient matrix, B is the constants vector

A = [
    [1, 1, 1],
    [2, 3, 1],
    [1, 2, 3]
]
B = [6, 14, 14]
n = len(B)  # Number of equations/variables

# Step 2: Forward Elimination
# Convert A into an upper triangular matrix
for i in range(n):  # Loop over each pivot row
    for k in range(i + 1, n):  # Loop over rows below pivot
        factor = A[k][i]  # Multiplier to eliminate element in column i
        for j in range(n):  # Loop over columns
            A[k][j] -= factor * A[i][j]  # Subtract pivot row * factor
        B[k] -= factor * B[i]  # Update corresponding element in B

print("Matrix A after forward elimination (upper triangular):")
print(A)
print("Modified B after forward elimination:")
print(B)

# Step 3: Back Substitution
# Solve for variables starting from the last row

x = [0] * n  # Initialize solution vector

for i in range(n - 1, -1, -1):  # Start from last row
    ax = 0
    for j in range(i + 1, n):  # Sum of known variables * coefficients
        ax += A[i][j] * x[j]
    x[i] = (B[i] - ax) / A[i][i]  # Solve for current variable

# Step 4: Print the solution
print("Solution vector [x, y, z]:")
print(x)




