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


                ###---------------TRAINING THE MULTIPLE LINEAR REGRESSION MODEL ON THE TRAINING SET--------------###


from sklearn.linear_model import LinearRegression

# NOTE : We didn't perform anything to solve the dummy variable trap or select the model(All In, Back Elimination etc, for selecting 
# the most significant column). This is bcoz the class that we imported from the module handles it itself.

lr = LinearRegression()
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)
np.set_printoptions(precision=3,suppress=True) # To set the decimal round off to 3 decimal places
print(np.concatenate(((y_pred.reshape(len(y_pred),1)),(y_test.reshape(len(y_test),1))),1)) # Comparing the predicted data and the actual
# data by concatinating the 2 vertically(made vertical using the .reshape() function) by the .concatenate function of the numpy module.
# We concatenated the 2 vectors bcoz we couldn't have plotted the graph in this case as there were many features in the FM.

# We can get the values of the coef and the intercepts using the same method in the simple LR model.


  ######################################## END OF THE MULTIPLE LINEAR REGRESSION MODEL ##################################################