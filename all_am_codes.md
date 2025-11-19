# Applied Mathematics Practical Codes - Complete Collection

## Practical No 1: Matrix Addition, Multiplication, and Scalar Operations

```python
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

print("Addition is:", add(a, b))
print("Multiplication is:", mul(a, b))
print(f"Scalar multiplication of 'a' with {scalar_value} is:", scalar_mul(a, scalar_value))
print(f"Scalar multiplication of 'b' with {scalar_value} is:", scalar_mul(b, scalar_value))
```

---

## Practical No 2: Compute Eigenvalues/Eigenvectors and Diagonalize a Matrix

```python
import numpy as np

# Step 1: Define 2x2 matrix
A = np.array([[4, 1],
              [2, 3]])

print("Matrix A:")
print(A)

# Step 2: Compute eigenvalues manually
a, b = A[0,0], A[0,1]
c, d = A[1,0], A[1,1]

trace = a + d
det = a*d - b*c

# Solve quadratic equation
lambda1 = (trace + np.sqrt(trace**2 - 4*det)) / 2
lambda2 = (trace - np.sqrt(trace**2 - 4*det)) / 2

print("\nEigenvalues:")
print(lambda1, lambda2)

# Step 3: Compute eigenvectors manually
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
    
    vec = np.array([v1, v2])
    return vec / np.linalg.norm(vec)

v1 = eigenvector(A, lambda1)
v2 = eigenvector(A, lambda2)

print("\nEigenvectors (normalized):")
print("v1 =", v1)
print("v2 =", v2)
```

---

## Practical No 3: Implement Gaussian Elimination from Scratch

```python
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
```

---

## Practical No 4: Discrete and Continuous Random Variables (Binomial and Normal Distributions)

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 1. BINOMIAL DISTRIBUTION
n = 10
p = 0.25 * 2
x_bino = np.arange(0, n + 1)
pmf_bino = stats.binom.pmf(x_bino, n, p)

# 2. NORMAL DISTRIBUTION
mu = 0
sigma = 1
x_norm = np.linspace(-4, 4, 100)
pmf_norm = stats.norm.pdf(x_norm, mu, sigma)

# 3. PLOTTING
plt.figure(figsize=(12, 5))

# Binomial Distribution
plt.subplot(1, 2, 1)
plt.stem(x_bino, pmf_bino, basefmt=" ")
plt.title("Binomial Distribution (n=10, p=0.5)")
plt.xlabel("Number of Successes")
plt.ylabel("Probability")
plt.grid(True)

# Normal Distribution
plt.subplot(1, 2, 2)
plt.plot(x_norm, pmf_norm, color="red")
plt.title("Normal Distribution (μ=0, σ=1)")
plt.xlabel("Value")
plt.ylabel("Probability Density")
plt.grid(True)

plt.tight_layout()
plt.show()
```

---

## Practical No 5: Implement Joint Probability Mass Function

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. Define outcomes of a single die
die_outcomes = [1, 2, 3, 4, 5, 6]

# 2. Construct sample space of 2 dice
sample_space = [(x, y) for x in die_outcomes for y in die_outcomes]

# 3. Joint PMF (all pairs equally likely)
joint_pmf = {}
for (x, y) in sample_space:
    joint_pmf[(x, y)] = 1 / 36

# 4. Store PMF values in a 6x6 matrix
pmf_matrix = np.zeros((6, 6))
for x in die_outcomes:
    for y in die_outcomes:
        pmf_matrix[x-1, y-1] = joint_pmf[(x, y)]

# 5. Visualization (Heatmap with values)
plt.figure(figsize=(7, 6))
plt.imshow(pmf_matrix, cmap="Blues", interpolation="nearest")

# Overlay probability values on each cell
for i in range(6):
    for j in range(6):
        plt.text(j, i, f"{pmf_matrix[i, j]:.3f}",
                ha="center", va="center", color="black")

plt.title("Joint PMF of Two 6-Sided Dice")
plt.xlabel("Outcome of Dice 2")
plt.ylabel("Outcome of Dice 1")
plt.colorbar(label="Probability")
plt.xticks(np.arange(6), die_outcomes)
plt.yticks(np.arange(6), die_outcomes)
plt.show()
```

---

## Practical No 6: Implement Central Limit Theorem and Plot Probability Distribution

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Central Limit Theorem Demonstration
sample_size = 100000
num_samples = 1000
sample_means = []
sample_std_devs = []

# 1. Generate samples from Uniform(0,1) and compute their mean & std. dev.
for _ in range(num_samples):
    sample = np.random.uniform(0, 1, sample_size)
    sample_means.append(np.mean(sample))
    sample_std_devs.append(np.std(sample))

# 2. Plot distribution of sample means
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
kde_means = stats.gaussian_kde(sample_means)
x_means = np.linspace(min(sample_means), max(sample_means), 100)
plt.plot(x_means, kde_means(x_means), color='g')
plt.title('Distribution of Sample Means (KDE)')
plt.xlabel('Sample Mean')
plt.ylabel('Density')
plt.grid(True)

# 3. Plot distribution of sample std devs
plt.subplot(1, 2, 2)
kde_std_devs = stats.gaussian_kde(sample_std_devs)
x_std_devs = np.linspace(min(sample_std_devs), max(sample_std_devs), 100)
plt.plot(x_std_devs, kde_std_devs(x_std_devs), color='b')
plt.title('Distribution of Sample Standard Deviations (KDE)')
plt.xlabel('Sample Standard Deviation')
plt.ylabel('Density')
plt.grid(True)

plt.tight_layout()
plt.show()
```

---

## Practical No 7: Generate t & F Distribution

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, f

# t-distribution
df_t = 10
x_t = np.linspace(-5, 5, 500)
pdf_t = t.pdf(x_t, df_t)

# F-distribution
dfn, dfd = 5, 20
x_f = np.linspace(0, 5, 500)
pdf_f = f.pdf(x_f, dfn, dfd)

# Plot both distributions
plt.figure(figsize=(12, 5))

# Plot t-distribution
plt.subplot(1, 2, 1)
plt.plot(x_t, pdf_t, 'g', label=f't-distribution (df={df_t})')
plt.title('t Distribution')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)

# Plot F-distribution
plt.subplot(1, 2, 2)
plt.plot(x_f, pdf_f, 'c', label=f'F-distribution (dfn={dfn}, dfd={dfd})')
plt.title('F Distribution')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

---

## Practical No 8: Perform Large Sample Test for Single Mean (Z-test)

```python
import numpy as np
from scipy.stats import norm

def z_test_single_mean(sample, mu0, sigma=None, alpha=0.05, alternative="two-sided"):
    sample = np.array(sample)
    n = len(sample)
    x_bar = np.mean(sample)
    
    if sigma is None:
        sigma = np.std(sample, ddof=1)
    
    # Compute Z-statistic
    z_stat = (x_bar - mu0) / (sigma / np.sqrt(n))
    
    # Compute p-value based on test type
    if alternative == "two-sided":
        p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    elif alternative == "greater":
        p_value = 1 - norm.cdf(z_stat)
    elif alternative == "less":
        p_value = norm.cdf(z_stat)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    
    reject = p_value <= alpha
    
    return {
        "Sample Mean": x_bar,
        "Hypothesized Mean": mu0,
        "Z-statistic": z_stat,
        "P-value": p_value,
        "Conclusion": "Reject H₀" if reject else "Fail to Reject H₀"
    }

# Example: Testing whether mean = 50
data = [52, 50, 53, 49, 48, 51, 54, 50, 52, 49]
mu0 = 50
alpha = 0.05

result = z_test_single_mean(data, mu0, alpha=alpha, sigma=None, alternative='two-sided')
print(result)
```

---

## Practical No 9: Calculate n-Step Transition Probability

```python
import numpy as np
import matplotlib.pyplot as plt

def validate_transition_matrix(P):
    P = np.array(P, dtype=float)
    
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square matrix")
    
    if np.any(P < 0):
        raise ValueError("Transition matrix cannot have negative entries.")
    
    if not np.allclose(P.sum(axis=1), 1.0):
        raise ValueError("Rows of P must sum to 1")
    
    return P

def n_step_transition_probability(P, n):
    P = validate_transition_matrix(P)
    return np.linalg.matrix_power(P, n)

# Main Program
if __name__ == "__main__":
    P = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.6, 0.3],
        [0.4, 0.2, 0.4]
    ])
    
    steps = [1, 2, 5, 10, 20]
    results = {}
    
    for n in steps:
        Pn = n_step_transition_probability(P, n)
        results[n] = Pn
        print(f"\n{n}-step transition probability matrix:\n{Pn}")
    
    state_names = ["S0", "S1", "S2"]
    max_n = 20
    
    evolution = [n_step_transition_probability(P, n)[0] for n in range(1, max_n + 1)]
    evolution = np.array(evolution)
    
    # Plot evolution of probabilities
    plt.figure(figsize=(8, 5))
    for i in range(P.shape[0]):
        plt.plot(range(1, max_n + 1), evolution[:, i], marker='o', label=f"P(S0→S{i})")
    
    plt.xlabel("Number of steps (n)")
    plt.ylabel("Probability")
    plt.title("n-step Transition Probabilities from S0 to Each State")
    plt.xticks(range(1, max_n + 1))
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
```

---

## Practical No 10: Simple Linear Regression

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Create the dataset
data = pd.DataFrame({
    'YearsExperience': [1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 3.0, 3.2, 3.2, 3.7, 3.9, 4.0, 4.0, 4.1, 4.5, 4.9,
                        5.1, 5.3, 5.9, 6.0, 6.8, 7.1, 7.9, 8.2, 8.7, 9.0, 9.5, 9.6, 10.3, 10.5, 11.0, 11.2],
    'Salary': [39343, 46205, 37731, 43525, 39891, 56642, 60150, 54445, 64445, 57189, 63218, 55794, 56957,
               57081, 61111, 67938, 66029, 83088, 81363, 93940, 91738, 98273, 101302, 113812, 109431,
               105582, 116969, 112635, 122391, 121872, 123000, 124000]
})

# Save to CSV
data.to_csv("Salary_dataset.csv", index=False)

# Prepare features and target
X = data[['YearsExperience']]
y = data['Salary']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R^2 Score:", r2)

# Visualize the results
plt.scatter(X_train, y_train, color='blue', label='Actual')
plt.plot(X_train, model.predict(X_train), color='red', label='Regression Line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary vs Years of Experience')
plt.legend()
plt.show()
```

---

## Practical No 11: Multiple Linear Regression

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Create Synthetic House Price Dataset
np.random.seed(42)
n = 120

# Generate features
size = np.random.randint(500, 4000, n)
bedrooms = np.random.randint(1, 6, n)
bathrooms = np.random.randint(1, 4, n)
distance = np.random.randint(1, 30, n)

# Create target variable (Price) with some random noise
price = (
    3000*bedrooms +
    5000*bathrooms +
    200*size -
    1500*distance +
    np.random.randint(-10000, 10000, n)
)

# Combine into a DataFrame
house_data = pd.DataFrame({
    "Size_sqft": size,
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "Distance_km": distance,
    "Price": price
})

# Save dataset to CSV
house_data.to_csv("house_prices.csv", index=False)
print("Dataset saved as house_prices.csv")
print(house_data.head())

# Step 2: Prepare Features and Target
X = house_data[["Size_sqft", "Bedrooms", "Bathrooms", "Distance_km"]]
y = house_data["Price"]

# Split the dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Make Predictions and Evaluate Model
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Coefficients (Weights):", model.coef_)
print("Intercept (β₀):", model.intercept_)
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

print("\nEquation:")
print(f"Price = {model.intercept_:.2f} "
      f"+ ({model.coef_[0]:.2f} * Size_sqft) "
      f"+ ({model.coef_[1]:.2f} * Bedrooms) "
      f"+ ({model.coef_[2]:.2f} * Bathrooms) "
      f"+ ({model.coef_[3]:.2f} * Distance_km)")

# Step 5: Visualization
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color="blue", alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices (Multiple Linear Regression)")
plt.show()

# Plot residuals
residuals = y_test - y_pred
plt.figure(figsize=(8,6))
plt.scatter(y_pred, residuals, color="green", alpha=0.6)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Price")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residuals vs Predicted Prices")
plt.show()
```

---

## Practical No 12: Implement Lagrange's Interpolation Method

```python
import numpy as np
import matplotlib.pyplot as plt

# Sample data points (x, y)
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 1, 3, 7])

def lagrange_interpolation(x_points, y_points, x_val):
    """
    x_points: array of x data points
    y_points: array of y data points
    x_val: the x-value at which interpolation is to be calculated
    Returns: interpolated y-value at x_val
    """
    n = len(x_points)
    result = 0.0
    
    # Iterate through each data point
    for i in range(n):
        # Compute the Lagrange basis polynomial L_i(x)
        term = y_points[i]
        for j in range(n):
            if i != j:
                term *= (x_val - x_points[j]) / (x_points[i] - x_points[j])
        result += term
    
    return result

# Interpolate at a new value
x_new = 2.5
y_new = lagrange_interpolation(x, y, x_new)
print(f"Interpolated value at x = {x_new} is y = {y_new}")

# Plotting the data points and interpolation curve
x_range = np.linspace(min(x), max(x), 100)
y_range = [lagrange_interpolation(x, y, xi) for xi in x_range]

plt.scatter(x, y, color='red', label='Data Points')
plt.plot(x_range, y_range, color='blue', label='Lagrange Interpolation')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Lagrange Interpolation")
plt.legend()
plt.grid(True)
plt.show()
```

---

## Practical No 13: Implement Newton's Interpolation Method

```python
def divided_diff(x, y):
    """
    Calculates the divided difference table for Newton's Interpolation.
    x: list of x values
    y: list of y values (f(x))
    returns: a 2D list representing the divided difference table
    """
    n = len(y)
    table = [[0 for _ in range(n)] for __ in range(n)]
    
    # First column is y values
    for i in range(n):
        table[i][0] = y[i]
    
    # Calculate divided differences
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i+1][j-1] - table[i][j-1]) / (x[i+j] - x[i])
    
    return table

def newton_interpolation(x, y, value):
    """
    Computes the interpolated value at 'value' using Newton's Interpolation.
    x: list of x values
    y: list of y values
    value: the point at which interpolation is required
    returns: interpolated value at x = value
    """
    n = len(x)
    table = divided_diff(x, y)
    
    # Start with first y value
    result = table[0][0]
    
    # Compute interpolation
    product_term = 1.0
    for i in range(1, n):
        product_term *= (value - x[i-1])
        result += table[0][i] * product_term
    
    return result

# Example usage:
x_points = [1, 2, 3, 4]
y_points = [1, 4, 9, 16]  # f(x) = x^2

# Interpolate at x = 2.5
value_to_interpolate = 2.5
interpolated_value = newton_interpolation(x_points, y_points, value_to_interpolate)

print(f"Interpolated value at x = {value_to_interpolate} is {interpolated_value}")
```

---

## How to Use These Codes

1. **Prerequisites**: Install required libraries:
   ```bash
   pip install numpy pandas matplotlib scipy scikit-learn
   ```

2. **Running the codes**: Copy any practical code and run it in your Python environment (Jupyter Notebook, VS Code, PyCharm, etc.)

3. **Notes**:
   - Each practical is self-contained and can be run independently
   - Some practicals generate CSV files (Practical 10 & 11)
   - All visualization codes will display plots automatically

---

**Date**: November 19, 2025  
**Course**: Applied Mathematics Practicals