import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('../Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Replace the Data with the Average of all the Data in that Column
# Here a Salary is missing so it is replaced by the average of all the Salaries in the Column
from sklearn.impute import SimpleImputer
# Call the Object and we want to replace all Values which are NaN
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# We only use the Average of the Numerical Values for replacement therefore we use th 'fit' Function
# So in the Features (x) we take all the Rows, but only the Column 1 and 2, the 3 is excluded 
imputer.fit(x[:, 1:3])
# Using the 'transform' function, we can put in the Values it return the updated Data 
x[:, 1:3] = imputer.transform(x[:, 1:3])

print(x)