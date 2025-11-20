import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset: Experience, Age -> Salary
X = np.array([
    [1, 22],
    [2, 25],
    [3, 28],
    [4, 32],
    [5, 35],
    [6, 38],
    [7, 40],
    [8, 44],
    [9, 47]
])

y = np.array([32, 45, 51, 63, 67, 76, 80, 88, 96])

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict
y_pred = model.predict(X)

print("Coefficients (b1, b2):", model.coef_)
print("Intercept (b0):", model.intercept_)

# Plot using first feature (Experience)
plt.scatter(X[:, 0], y, color='blue', label="Actual Data")
plt.plot(X[:, 0], y_pred, color='red', label="Regression Line", linewidth=2)
plt.xlabel("Years of Experience")
plt.ylabel("Salary (in thousands)")
plt.title("Multiple Linear Regression")
plt.legend()
plt.grid(True)
plt.show()
