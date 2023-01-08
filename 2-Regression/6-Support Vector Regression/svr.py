# SVR

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)

#fitting SVR to dataset
from sklearn.svm import SVR
#kernel --> linear for linear model and --> poly, rbf for nonlinear model
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

#predicting a new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#Visualing the SVR result
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth and Bluff (SVR)')
plt.xlabel('Position of level')
plt.xlabel('Salary')
plt.show()


#Visualing the SVR result (higher resolution and smooth curve)
X_grid = np.arange(min(X), max(X), 0.1) 
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth and Bluff (SVR)')
plt.xlabel('Position of level')
plt.xlabel('Salary')
plt.show()