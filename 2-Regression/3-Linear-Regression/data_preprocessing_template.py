#Data Preprocessing Template

#Import Library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset = pd.read_csv('Salary_Data.csv')

#Create Matrix Feature
X = dataset.iloc[:, :-1].values

#Create Dependeny Varriable
y = dataset.iloc[:, 1].values

#Spliting Data into Training Set and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Featuring Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""