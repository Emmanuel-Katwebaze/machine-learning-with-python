# Self Organizing Map for Fraud Detection

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
#frauds = np.concatenate((mappings[(8, 1)], mappings[(6, 8)]), axis = 0)
frauds = np.concatenate((mappings[(6, 4)], mappings[(7, 5)]), axis = 0)
frauds = sc.inverse_transform(frauds)

#Printing the fraud clients

#print('Fraud Customer IDs')
#for i in frauds[:, 0]:
#  print(int(i))
