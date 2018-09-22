import numpy as np
import os
os.chdir("\\Users\sbayed\Desktop\sbayed_local\Machine Learning\AI\AI")
from DFF import *
from utilities import *

X_train, Y_train = load_dataset()

layers_dims = [X_train.shape[0], 5, 2, 1]
activations = ['none','relu','relu','sigmoid']
cost = 'cross-entropy'
lambd = 0

parameters = model(X_train, Y_train, layers_dims, activations, cost, lambd, learning_rate = 0.0007, num_epochs = 10000, mini_batch_size = 64, print_cost = True)

# Validation
ACC_train = pred_train = predict(X_train, Y_train, parameters)

