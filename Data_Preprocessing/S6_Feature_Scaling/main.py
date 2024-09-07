import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('../Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)


# Use as a Feature Scaling technique the Standardisation
from sklearn.preprocessing import StandardScaler
# Call the Object of the Class StandardScaler
sc = StandardScaler()
# The Fit and Transform Method of the Standard Scaler will use the Standardisation Formula to transform
# Standardistation Formula =  (X - mean(X) / standard deviation(X))
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
# Only use Transform methode, because if we use the Fit method we gonna get a new Scaler, which dosen't makes any sense
x_test[:, 3:] = sc.transform(x_test[:, 3:])

print(x_train)
print(x_test)