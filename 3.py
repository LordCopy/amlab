# Practical 3: Implement Gaussian Elimination from Scratch

# Step 1: Define the system of equations
A = [
    [1, 1, 1],
    [2, 3, 1],
    [1, 2, 3]
]
B = [6, 14, 14]

n = len(B)

# Step 2: Forward Elimination
for i in range(n):
    for k in range(i+1, n):
        factor = A[k][i] / A[i][i]
        for j in range(i, n):
            A[k][j] -= factor * A[i][j]
        B[k] -= factor * B[i]

print("Matrix A after forward elimination (upper triangular):")
for row in A:
    print(row)
print("Modified B after forward elimination:")
print(B)

# Step 3: Back Substitution
x = [0] * n
for i in range(n-1, -1, -1):
    ax = 0
    for j in range(i+1, n):
        ax += A[i][j] * x[j]
    x[i] = (B[i] - ax) / A[i][i]

print("Solution vector [x, y, z]:")
print(x)
