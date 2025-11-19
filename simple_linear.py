import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Practical 10 — Simple linear regression (Salary vs Experience)

def run_linear_regression():
    data = pd.DataFrame({
        'YearsExperience': [1.1,1.3,1.5,2.0,2.2,2.9,3.0,3.2,3.2,3.7,3.9,4.0,4.0,4.1,4.5,4.9,5.1,5.3,5.9,6.0,6.8,7.1,7.9,8.2,8.7,9.0,9.5,9.6,10.3,10.5,11.0,11.2],
        'Salary': [39343,46205,37731,43525,39891,56642,60150,54445,64445,57189,63218,55794,56957,57081,61111,67938,66029,83088,81363,93940,91738,98273,101302,113812,109431,105582,116969,112635,122391,121872,123000,124000]
    })
    data.to_csv('Salary_dataset.csv', index=False)
    X = data[['YearsExperience']]
    y = data['Salary']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = LinearRegression(); model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print('MSE:', mean_squared_error(y_test,y_pred))
    print('R2:', r2_score(y_test,y_pred))
    plt.scatter(X_train,y_train); plt.plot(X_train, model.predict(X_train), color='red'); plt.show()

if __name__ == '__main__':
    run_linear_regression()
