import numpy as np
from DFF import *
from utilities import *

X_train_orig, Y_train, X_test_orig, Y_test, classes = load_data()
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T  
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

X_train = X_train_flatten/255.
X_test = X_test_flatten/255.

