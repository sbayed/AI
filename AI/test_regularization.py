import numpy as np
import os
os.chdir("\\Users\sbayed\Desktop\sbayed_local\Machine Learning\AI\AI")
from DFF import *
from utilities import *

X_train, Y_train, X_test, Y_test = load_2D_dataset()


# Inputs
layers_dims = [X_train.shape[0], 20, 3, 1]
activations = ['none','relu','relu','sigmoid']
cost = 'cross-entropy'
lambd = 0.7

# Esimation
parameters = L_layer_model(X_train, Y_train, layers_dims, activations, cost, lambd, learning_rate = 0.3, num_iterations = 1000, print_cost = True)


# Validation
ACC_train = pred_train = predict(X_train, Y_train, parameters)
ACC_test = pred_test = predict(X_test, Y_test, parameters)

