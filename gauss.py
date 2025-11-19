# Practical 3 — Gaussian elimination (solve linear system)

A = [
    [1, 1, 1],
    [2, 3, 1],
    [1, 2, 3]
]
B = [6, 14, 14]

def solve_gauss(A, B):
    n = len(B)
    # forward elimination
    for i in range(n):
        for k in range(i+1, n):
            factor = A[k][i] / A[i][i]
            for j in range(i, n):
                A[k][j] -= factor * A[i][j]
            B[k] -= factor * B[i]
    # back substitution
    x = [0]*n
    for i in range(n-1, -1, -1):
        s = 0
        for j in range(i+1, n):
            s += A[i][j]*x[j]
        x[i] = (B[i] - s)/A[i][i]
    return x

if __name__ == '__main__':
    sol = solve_gauss([row[:] for row in A], B[:])
    print("Solution [x,y,z]:", sol)
