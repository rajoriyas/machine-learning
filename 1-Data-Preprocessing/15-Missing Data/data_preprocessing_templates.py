#Data Preprocessing

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Impororting the dataset
dataset = pd.read_csv('Data.csv')

#Create the matrix of features
X = dataset.iloc[:, :-1].values

#Create dependent varriable
y = dataset.iloc[:, 3].values

#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
df_X = pd.DataFrame(X)
