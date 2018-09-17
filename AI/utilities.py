import matplotlib.pyplot as plt
import h5py
import numpy as np
import scipy.io


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
def dictionary_to_vector(parameters):
    """
    Roll all our parameters dictionary into a single vector satisfying our specific required shape.
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
def vector_to_dictionary(theta, layers_dims, activations):
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