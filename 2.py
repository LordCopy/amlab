import numpy as np

# Practical 2: Compute Eigenvalues/Eigenvectors and Diagonalize a Matrix

def eigen_manual(A):
    # compute eigenvalues for 2x2 matrix A
    a, b = A[0,0], A[0,1]
    c, d = A[1,0], A[1,1]

    trace = a + d
    det = a*d - b*c

    lambda1 = (trace + np.sqrt(trace**2 - 4*det)) / 2
    lambda2 = (trace - np.sqrt(trace**2 - 4*det)) / 2

    return lambda1, lambda2

def eigenvector(A, lam):
    a, b = A[0,0], A[0,1]
    c, d = A[1,0], A[1,1]
    if b != 0:
        v1 = 1
        v2 = -((a - lam)/b)*v1
    elif c != 0:
        v2 = 1
        v1 = -((d - lam)/c)*v2
    else:
        v1, v2 = 1, 0

    vec = np.array([v1, v2], dtype=float)
    return vec / np.linalg.norm(vec)

if __name__ == '__main__':
    A = np.array([[4, 1],
                  [2, 3]], dtype=float)

    print("Matrix A:")
    print(A)

    lambda1, lambda2 = eigen_manual(A)
    print("\nEigenvalues:")
    print(lambda1, lambda2)

    v1 = eigenvector(A, lambda1)
    v2 = eigenvector(A, lambda2)
    print("\nEigenvectors (normalized):")
    print("v1 =", v1)
    print("v2 =", v2)
