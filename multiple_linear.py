# importing libraries
import numpy as np
import pandas as pd

# importing the dataset and dividing into training and test set.
ds = pd.read_csv('50_Startups2.csv')
x = ds.iloc[:,:-1].values
y = ds.iloc[:,-1].values

# As there is no missing data in the dataset we can move on to split the set into training and test set.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=4)

# Encoding the categorical data in the matrix of features

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')

x_train = ct.fit_transform(x_train)
x_test = ct.fit_transform(x_test)

# in the dataset all the values except for the city are in the units of money so feature scaling not required as the scale is same.
