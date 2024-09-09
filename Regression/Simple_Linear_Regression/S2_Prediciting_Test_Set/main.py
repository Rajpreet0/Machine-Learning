import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('../Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test Set Results

# Call the predict function to predict the Salary based on the Years
y_pred = regressor.predict(X_test)
print(y_pred)