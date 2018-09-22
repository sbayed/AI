import matplotlib.pyplot as plt
import h5py
import numpy as np
import scipy.io
import math
import sklearn
import sklearn.datasets


def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
def print_mislabeled_images(classes, X, y, p):
    """
    Plots images where predictions and truth were different.
    X -- dataset
    y -- true labels
    p -- predictions
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]
        
        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))
def load_2D_dataset():
    data = scipy.io.loadmat('datasets/data.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T


    return train_X, train_Y, test_X, test_Y
def load_dataset():
    np.random.seed(3)
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2) #300 #0.2 
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    
    return train_X, train_Y
def parameters_to_vector(parameters):
    """
    Roll all parameters dictionary into a single vector satisfying specific required shape for gradient checking.
    """
    layers_dims = []
    activations = ['none']
    L = len(parameters) // 3
    count = 0
    for l in range(L):

        W = parameters["W" + str(l+1)]
        # flatten parameter
        new_vector = np.reshape(W, (-1, 1))

        if count == 0:
            theta = new_vector
            layers_dims.append(W.shape[1])
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

        b = parameters["b" + str(l+1)]
        # flatten parameter
        new_vector = np.reshape(b, (-1, 1))
        theta = np.concatenate((theta, new_vector), axis=0)
        layers_dims.append(W.shape[0])
        activations.append(parameters['g' + str(l+1)])

    cache = (layers_dims, activations)


    return theta, cache
def vector_to_parameters(theta, layers_dims, activations):
    """
    Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
    """
    parameters = {}

    L = len(layers_dims)
    idx = 0

    for l in range(1, L):
        n_w = layers_dims[l]*layers_dims[l-1]
        n_b = layers_dims[l]
        n = n_w + n_b
        print(idx)
        Wb_vec = theta[idx:idx + n]
        W_vec = Wb_vec[:n_w]
        b_vec = Wb_vec[n_w:n]
        parameters['W' + str(l)] = W_vec.reshape((layers_dims[l], layers_dims[l-1]))
        parameters['b' + str(l)] = b_vec.reshape((layers_dims[l], 1))
        parameters['g' + str(l)] = activations[l]
        idx = idx + n



    return parameters
def gradients_to_vector(gradients):
    """
    Roll all gradients dictionary into a single vector satisfying specific required shape for gradient checking.
    """
    L = len(gradients) // 3
    count = 0
    for l in range(L):
        dW = gradients["dW" + str(l+1)]
        # flatten gradient
        new_vector = np.reshape(dW, (-1, 1))

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1
        count = count + 1

        db = gradients["db" + str(l+1)]
        # flatten gradient
        new_vector = np.reshape(db, (-1, 1))
        theta = np.concatenate((theta, new_vector), axis=0)

    return theta
def gradient_check(parameters, gradients, X, Y, cost, lambd, epsilon=1e-7):
    """
    Checks if backward propagation computes correctly the gradient of the cost output by forward propagation

    Arguments:
    parameters -- python dictionary containing the weights
    grad -- output of backward propagation, contains gradients of the cost with respect to the weights.
    X -- input matrix
    Y -- output matrix
    epsilon -- shift to the input to compute approximated gradient

    Returns:
    difference -- difference between the approximated gradient and the backward propagation gradient
    """

    # Set-up variables
    parameters_values, cache = parameters_to_vector(parameters)
    layers_dims, activations = cache
    grads_values = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))

    # Compute gradapprox
    for i in range(num_parameters):
        thetaplus = np.copy(parameters_values)
        thetaplus[i][0] = thetaplus[i][0] + epsilon
        AL, _ = L_model_forward(X, vector_to_parameters(thetaplus, layers_dims, activations))
        J_plus[i] = compute_cost(AL, Y, cost, vector_to_parameters(thetaplus, layers_dims, activations), lambd)

        thetaminus = np.copy(parameters_values)
        thetaminus[i][0] = thetaminus[i][0] - epsilon
        AL, _ = L_model_forward(X, vector_to_parameters(thetaminus, layers_dims, activations))
        J_minus[i] = compute_cost(AL, Y, cost, vector_to_parameters(thetaminus, layers_dims, activations), lambd)

        # Compute gradapprox[i]
        gradapprox[i] = (J_plus[i] - J_minus[i])/(2*epsilon)

    # Compare gradapprox to backward propagation gradients by computing difference.
    numerator = np.linalg.norm(grads_values - gradapprox)
    denominator = np.linalg.norm(grads_values) + np.linalg.norm(gradapprox) 
    difference = numerator/denominator

    if difference > 2e-7:
        print(
            "\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print(
            "\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")

    return difference
def random_mini_batches(X, Y, mini_batch_size = 64):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input matrix
    Y -- output matrix
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """        
    m = X.shape[1]                  
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) 
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:,k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:,(k+1)*mini_batch_size :]
        mini_batch_Y = shuffled_Y[:,(k+1)*mini_batch_size :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

