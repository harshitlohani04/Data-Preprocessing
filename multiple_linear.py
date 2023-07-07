# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the dataset and dividing into training and test set.
ds = pd.read_csv('50_Startups.csv')
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

#Applying feature scaling on the remaining columns

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:,3:] = sc.fit_transform(x_train[:,3:])
x_test[:,3:] = sc.transform(x_test[:,3:])
print(x_train,'\n\n',x_test)

               
