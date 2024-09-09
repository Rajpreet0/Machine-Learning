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

y_pred = regressor.predict(X_test)

# Visualize using 2D Plot - Training Set

# scatter allows us to put the real salaries (y_train) into a 2D plot (it accepts coordinates)
plt.scatter(X_train, y_train, color='red')

# Plot the regression line using the Plot Method
plt.plot(X_train, regressor.predict(X_train), color='blue') # Remember we are Visualizing the Training Set so we still have to predict X_train

# Design the Plot with Labels and Title
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

plt.show()


# Visualize using 2D Plot - Test Set

# scatter allows us to put the real salaries (y_train) into a 2D plot (it accepts coordinates)
plt.scatter(X_test, y_test, color='red')

# Plot the regression line using the Plot Method
plt.plot(X_train, regressor.predict(X_train), color='blue')

# Design the Plot with Labels and Title
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

plt.show()