import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('../Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Training the Simple Linear Regression model on the Training Set

from sklearn.linear_model import LinearRegression
# Create an instance of the Class LinearRegression which is the LinearRegression Model
regressor = LinearRegression()
# Train the Model using the Fit method on the training set
regressor.fit(X_train, y_train)

