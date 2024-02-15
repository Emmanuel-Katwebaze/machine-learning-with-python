# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 21:05:50 2024

@author: Emma
"""

# Make a Hybrid Deep Learning Model

# Part 1 - Identify the Frauds with the Self-Organizing Map
# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Train the SOM
from minisom import MiniSom
# x and y are the dimensions of the SOM, input_len = number of columns, learning_rate = is the rate at which the weights are updated
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T) # returns the MID in one matrix and the .T takes the transpose of the MID matrix
colorbar()
markers = ['o', 's'] # o - circle, s - square
colors = ['r', 'g'] # r - red, g - green
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5, 
         markers[y[i]], 
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None', 
         markersize = 10, 
         markeredgewidth = 2) # placing the marker at the center of the square corresponding to the winning node
show()

# Finding the frauds
mappings = som.win_map(X)
#choosing the mappings is very arbitrary
frauds = np.concatenate((mappings[(7, 5)], mappings[(5, 1)]), axis = 0)
frauds = sc.inverse_transform(frauds)


# Part 2 - Going from Unsupervised to Supervised Deep Learning
# Creating the matrix of features
customers = dataset.iloc[:, 1:].values # customer id not necessary but last column i.e. if got loan or not is necessary

# Creating the dependent variable
is_fraud = np.zeros(len(dataset)) # generate a group of zeros which will later be given ones
for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds: # dataset.iloc[i, 0] is customer id
        is_fraud[i] = 1
 
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 2, kernel_initializer='uniform', activation = 'relu', input_dim = 15 ))


# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer='uniform', activation = 'sigmoid' ))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(customers, is_fraud, batch_size = 1, epochs = 2)


# Predicting the probability of frauds
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)
y_pred = y_pred[y_pred[:, 1].argsort()] # sort your numpy array by the column of index 1