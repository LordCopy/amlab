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
