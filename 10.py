import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# More realistic dataset (Years of Exp vs Salary with variation)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
y = np.array([32, 45, 51, 63, 67, 76, 80, 88, 96])  # not perfectly straight

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict
y_pred = model.predict(X)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Plot
plt.scatter(X, y, color='blue', label="Data Points")
plt.plot(X, y_pred, color='red', label="Regression Line", linewidth=2)
plt.xlabel("Years of Experience")
plt.ylabel("Salary (in thousands)")
plt.title("Simple Linear Regression (More Realistic Data)")
plt.grid(True)
plt.legend()
plt.show()
