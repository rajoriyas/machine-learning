#Data Preprocessing

#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

# Load data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

Y_train = train["label"]
X_train = train.drop(labels = ["label"], axis = 1) 

# Check for null and missing values
X_train.isnull().any().describe()
test.isnull().any().describe()

# We perform a grayscale normalization to reduce the effect of illumination's differences.
# Moreover the CNN converg faster on [0..1] data than on [0..255].
# Normalize the data
X_train = X_train / 255.0
test = test / 255.0

# Reshape data
X_train = X_train.values.reshape(-1, 28, 28, 1) # -1, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS
test = test.values.reshape(-1, 28, 28, 1) # -1, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS

# One-Hot encoding
Y_train = to_categorical(Y_train, num_classes=10)

# Split the train and the validation set for the fitting
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)

# del X_val
# del Y_val

# Some examples
g = plt.imshow(X_train[1][:,:,0])

# Convolutional Neural Network

# Part 1 - Building the CNN

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, 3, 3, input_shape = (28, 28, 1), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPool2D(pool_size = (2, 2)))

# Step 2.A - Improve Accuracy
classifier.add(Conv2D(64, 3, 3, activation = 'relu'))
classifier.add(MaxPool2D(pool_size = (2, 2)))

# Adding a second convolutional layer

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 10, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy']) #categorial_crossentropy

epochs = 1 # Turn epochs to 30 to get 0.9967 accuracy
batch_size = 86

history = classifier.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_test, Y_test), verbose = 2)



