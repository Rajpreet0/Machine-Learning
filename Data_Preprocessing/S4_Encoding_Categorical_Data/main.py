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

# You don't want to have Strings in your Dataset especially, when there are multiple strings with the same name
# The Model could get confused, therfore you want to change that into numerical values (binary values)
from sklearn.compose import ColumnTransformer
# First we have  France, Spain, Germany you spilt this into a Vector so three Columns
# France would be [1,0,0] - Spain would be [0,1,0] - Germany would be [0,0,1]
from sklearn.preprocessing import OneHotEncoder
# Create an Object of the ColumnTransformer Class
# The ColumnTransformer Class expects us to say what we want to transform and what it should do with the Columns which are not effected
# of the transformer Argument it needs the method (encoding) then the what encoding Method (OneHotEncoder) and which Column (the First)
# in the remainder we use 'passthrough' meaning, that we want to keep the remainder columns
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# The ColumnTransformer Class has a Method for fitting and transforming all at once
x = np.array(ct.fit_transform(x))

# LabelEncoder Class is used to transform Yes, No to 1 and 0
from sklearn.preprocessing import LabelEncoder
# Call the an Object of the LabelEncoder Class
le = LabelEncoder()
# Convert the Column of Yes and No's to 1 and 0`s
y = le.fit_transform(y)

print(x)
print(y)