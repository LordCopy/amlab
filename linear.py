import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Practical 11 — Multiple linear regression (synthetic house prices)

def run_multiple_regression():
    np.random.seed(42)
    n = 120
    size = np.random.randint(500,4000,n)
    bedrooms = np.random.randint(1,6,n)
    bathrooms = np.random.randint(1,4,n)
    distance = np.random.randint(1,30,n)
    price = 3000*bedrooms + 5000*bathrooms + 200*size - 1500*distance + np.random.randint(-10000,10000,n)
    house_data = pd.DataFrame({
        'Size_sqft': size,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Distance_km': distance,
        'Price': price
    })
    house_data.to_csv('house_prices.csv', index=False)
    X = house_data[['Size_sqft','Bedrooms','Bathrooms','Distance_km']]
    y = house_data['Price']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = LinearRegression(); model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print('Coefficients:', model.coef_) 
    print('Intercept:', model.intercept_)
    print('MSE:', mean_squared_error(y_test,y_pred))
    print('R2:', r2_score(y_test,y_pred))
    plt.scatter(y_test,y_pred); plt.plot([y.min(),y.max()],[y.min(),y.max()], 'r--'); plt.show()

if __name__ == '__main__':
    run_multiple_regression()
