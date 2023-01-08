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

#Split Training Set and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""