import numpy as np
from DFF import *
from utilities import *

X_train_orig, Y_train, X_test_orig, Y_test, classes = load_data()
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T  
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

X_train = X_train_flatten/255.
X_test = X_test_flatten/255.

# Inputs

layers_dims = [12288, 20, 7, 5, 1]

# Esimation

parameters = L_layer_model(X_train, Y_train, layers_dims, num_iterations = 2500, print_cost = True)


# Validation

pred_train = predict(X_train, Y_train, parameters)
pred_test = predict(X_test, Y_test, parameters)