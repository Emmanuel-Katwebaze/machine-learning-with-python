# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 09:53:02 2024

@author: Emma
"""
##Downloading the dataset

###ML-100K
"""

!wget "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
!unzip ml-100k.zip
!ls

"""###ML-1M"""

!wget "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
!unzip ml-1m.zip
!ls


# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t')
training_set = np.array(training_set, dtype = 'int') # convert the df into an array because tensors take in arrays
test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

# Converting the data into an array with users in lines and movies in columns

def convert(data):
  new_data = []
  for id_users in range(1, nb_users + 1):
    id_movies = data[:, 1] [data[:, 0] == id_users]
    id_ratings = data[:, 2] [data[:, 0] == id_users]
    ratings = np.zeros(nb_movies)
    ratings[id_movies - 1] = id_ratings # -1 because indices start at 0 in the ratings list but id_movies start at 1
    new_data.append(list(ratings))
  return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors

training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network - Stacked Auto Encoders
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20) # fc - full connection, 20 nodes in first hidden layer (it's experimental and can be tuned), 
        self.fc2 = nn.Linear(20, 10) # 20 - 1st layer, 10 nodes in second hidden layer (it's experimental and can be tuned),
        self.fc3 = nn.Linear(10, 20) # 10 - 2nd layer, 20 in the 3rd layer
        self.fc4 = nn.Linear(20, nb_movies) # output layer should have same number of layers in the input layer
        self.activation = nn.Sigmoid()
    def forward(self, x): # here x is input vector of features
        x = self.activation(self.fc1(x)) 
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x)) # carries out the decoding
        x = self.fc4(x) # final part of decoding
        return x # vector of predicted ratings to compare to the real ratings and compare the loss
sae = SAE()
criterion = nn.MSELoss()
# optimizer to carry out stochastic gradient descent, RMSprop proved better than Adam
# lr - learning rate (experimental), weight_decay - used to reduce the learning rate after every epoch and hence regulates the convergence
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

# Training the SAE
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0) # creates a batch of a single input vector
        target = input.clone()
        if torch.sum(target.data > 0) > 0: # checking if user rated at least 1 movie, helps save memory
            output = sae.forward(input)
            target.require_grad = False # makes sure we don't compute the gradient with respect to the target, saving a lot of memory
            output[target == 0] = 0 # won't impact the updating of the weights, saves memory
            loss = criterion(output, target) # output - vector of predicted ratings, target - vector of real ratings
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e - 10) # 1e - 10 makes sure the denominator is not equal to zero as a denominator and prevents infinite error
          s += 1.
          optimizer.step() # updates the weights i.e. decides the intensity to which the weights will be updated i.e. the amount
      print('epoch: '+str(epoch)+'loss: '+ str(train_loss/s)) # train_loss/s - getting the average train loss

# Testing the SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
  input = Variable(training_set[id_user]).unsqueeze(0)
  target = Variable(test_set[id_user]).unsqueeze(0)
  if torch.sum(target.data > 0) > 0:
    output = sae(input)
    target.require_grad = False
    output[target == 0] = 0
    loss = criterion(output, target)
    mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
    test_loss += np.sqrt(loss.data*mean_corrector)
    s += 1.
print('test loss: '+str(test_loss/s))