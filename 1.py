a = [[7, 3], [2, 8]]
b = [[4, 6], [9, 2]]
scalar = 7

def add(x, y):
    result = [[0 for _ in range(len(x[0]))] for _ in range(len(x))]
    for i in range(len(x)):
        for j in range(len(x[0])):
            result[i][j] = x[i][j] + y[i][j]
    return result

def mul(A, B):
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
            sum = 0
            for k in range(len(A[0])):
                sum += A[i][k] * B[k][j]
            row.append(sum)
        result.append(row)
    return result

def scalar_mul(matrix, scalar):
    result = []
    for i in range(len(matrix)):
        row = [matrix[i][j] * scalar for j in range(len(matrix[0]))]
        result.append(row)
    return result

print("Addition:", add(a, b))
print("Multiplication:", mul(a, b))
print("Scalar A:", scalar_mul(a, scalar))
print("Scalar B:", scalar_mul(b, scalar))



import numpy as np

a = np.array([[7, 3],
              [2, 8]])

b = np.array([[4, 6],
              [9, 2]])

scalar = 7

add_result = a + b
mul_result = np.dot(a, b)    
scalar_a = scalar * a
scalar_b = scalar * b
print("Addition:\n", add_result)
print("\nMultiplication:\n", mul_result)
print(f"\nScalar multiplication of A with {scalar}:\n", scalar_a)
print(f"\nScalar multiplication of B with {scalar}:\n", scalar_b)

