# Practical 1: Matrix Addition, Multiplication, and Scalar Operations

# Define two matrices 'a' and 'b'
a = [[7, 3],
     [2, 8]]
b = [[4, 6],
     [9, 2]]

# Define a scalar value
scalar_value = 7

# Matrix Addition Function
def add(x, y):
    result = [[0 for _ in range(len(x[0]))] for _ in range(len(x))]
    for i in range(len(x)):
        for j in range(len(x[0])):
            result[i][j] = x[i][j] + y[i][j]
    return result

# Matrix Multiplication Function
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

# Scalar Multiplication Function
def scalar_mul(matrix, scalar):
    result = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix[0])):
            row.append(matrix[i][j] * scalar)
        result.append(row)
    return result

if __name__ == '__main__':
    print("Addition is:", add(a, b))
    print("Multiplication is:", mul(a, b))
    print(f"Scalar multiplication of 'a' with {scalar_value} is:", scalar_mul(a, scalar_value))
    print(f"Scalar multiplication of 'b' with {scalar_value} is:", scalar_mul(b, scalar_value))
