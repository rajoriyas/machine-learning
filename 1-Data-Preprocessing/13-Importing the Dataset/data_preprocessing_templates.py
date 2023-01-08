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
