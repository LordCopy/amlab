import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Practical 11: Multiple Linear Regression

def run_multiple_regression():
    np.random.seed(42)
    n = 120

    size = np.random.randint(500, 4000, n)
    bedrooms = np.random.randint(1, 6, n)
    bathrooms = np.random.randint(1, 4, n)
    distance = np.random.randint(1, 30, n)

    price = (
        3000*bedrooms +
        5000*bathrooms +
        200*size -
        1500*distance +
        np.random.randint(-10000, 10000, n)
    )

    house_data = pd.DataFrame({
        "Size_sqft": size,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Distance_km": distance,
        "Price": price
    })

    house_data.to_csv("house_prices.csv", index=False)
    print("Dataset saved as house_prices.csv")

    X = house_data[["Size_sqft", "Bedrooms", "Bathrooms", "Distance_km"]]
    y = house_data["Price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

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

    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, color="blue", alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel("Actual House Price")
    plt.ylabel("Predicted House Price")
    plt.title("Actual vs Predicted House Prices (Multiple Linear Regression)")
    plt.show()

    residuals = y_test - y_pred
    plt.figure(figsize=(8,6))
    plt.scatter(y_pred, residuals, color="green", alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Residuals vs Predicted Prices")
    plt.show()

if __name__ == '__main__':
    run_multiple_regression()
