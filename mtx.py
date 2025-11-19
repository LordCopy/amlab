# Practical 1 — Matrix operations: addition, multiplication, scalar multiplication

a = [[7, 3],
     [2, 8]]
b = [[4, 6],
     [9, 2]]

scalar_value = 7

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
            s = 0
            for k in range(len(A[0])):
                s += A[i][k] * B[k][j]
            row.append(s)
        result.append(row)
    return result

def scalar_mul(matrix, scalar):
    result = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix[0])):
            row.append(matrix[i][j] * scalar)
        result.append(row)
    return result

if __name__ == '__main__':
    print("Addition:", add(a, b))
    print("Multiplication:", mul(a, b))
    print(f"Scalar multiply a by {scalar_value}:", scalar_mul(a, scalar_value))
