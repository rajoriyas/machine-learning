# Polynomial Regression

#Data preprocessing template
#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import dataset
dataset = pd.read_csv('Position_Salaries.csv')
#X = dataset.iloc[:, 1].values
"""To make it a matrix"""
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

#Spliting the dataset into Training set and Test set
"""due to small number of dataset"""
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

#Fitting Linear Regression to dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Fitting Polynomial Regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) # use 2,3,4...
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#Visualising Linear Regression Result
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualising Polynomial Regression Result
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
#We don't put X_poly because graph is between X and fxn(X) not X and fxn(X_poly)
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Predicting a new result using linear regression
lin_reg.predict([[6.5]])

#Predicting a new result using polynomial regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))