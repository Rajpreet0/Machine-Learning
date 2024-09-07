import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the CSV File using Pandas to create a Data Frame
dataset = pd.read_csv('../Data.csv')
# Features in the Dataset without the Dependent Variable Vector
# iloc to use Indexes [Row, Column] -> : is a range (without upper, lower bound => take all)
# -1 means the last column, which is excluded here
x = dataset.iloc[:, :-1].values
# Dependent Variable Vector without the Features
# We only want to have the last column so NO range
y = dataset.iloc[:, -1].values

print(x)
print(y)