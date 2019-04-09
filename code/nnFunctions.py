import numpy as np
from scipy.optimize import minimize
from math import sqrt
import pickle
'''
You need to modify the functions except for initializeWeights() and preprocess()
'''

def initializeWeights(n_in, n_out):
    '''
    initializeWeights return the random weights for Neural Network given the
    number of node in the input layer and output layer
    Input:
    n_in: number of nodes of the input layer
    n_out: number of nodes of the output layer
    Output:
    W: matrix of random initial weights with size (n_out x (n_in + 1))
    '''
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def preprocess(filename,scale=True):
    '''
     Input:
     filename: pickle file containing the data_size
     scale: scale data to [0,1] (default = True)
     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    '''
    with open(filename, 'rb') as f:
        train_data = pickle.load(f)
        train_label = pickle.load(f)
        test_data = pickle.load(f)
        test_label = pickle.load(f)
    # convert data to double
    train_data = train_data.astype(float)
    test_data = test_data.astype(float)

    # scale data to [0,1]
    if scale:
        train_data = train_data/255
        test_data = test_data/255

    return train_data, train_label, test_data, test_label

def sigmoid(z):
    '''
    Notice that z can be a scalar, a vector or a matrix
    return the sigmoid of input z (same dimensions as z)
    '''
    # your code here - remove the next four lines
    s = 1/(1 + np.exp(-z))
    return s

def nnObjFunction(params, *args):
    '''
    % nnObjFunction computes the value of objective function (cross-entropy
    % with regularization) given the weights and the training data and lambda
    % - regularization hyper-parameter.
    % Input:
    % params: vector of weights of 2 matrices W1 (weights of connections from
    %     input layer to hidden layer) and W2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not including the bias node)
    % n_hidden: number of node in hidden layer (not including the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % train_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % train_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector (not a matrix) of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.
    '''
    n_input, n_hidden, n_class, train_data, train_label, lambdaval = args
    # First reshape 'params' vector into 2 matrices of weights W1 and W2
    W1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1))) # W1 = 3 x 6
    W2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1))) # W2 = 2 x 4
    obj_val = 0 # obj_val = scalar
    # Your code here
    #
    #
    #
    #
    #
    true_input = np.zeros((np.size(train_label), n_class)) # true_input = 2 x 2
    train_data = np.insert(train_data, train_data.shape[1], 1, axis = 1) # train_data = 2 x 6
    size_train_data = train_data.shape[0] # size_train_data = scalar
    grad_W1 = 0 # obj_val = scalar
    grad_W2 = 0 # obj_val = scalar
    error = 0 # obj_val = scalar

    # 1 of k encoding to model the categorical data
    for i in range(np.size(train_label)):
        # 1 of k encoding to model the categorical data
        input_label = train_label[i]
        true_input[i][input_label] = 1
    true_input = true_input.transpose()
    
    # performing a forward pass
    out_hidden, out_final = forwardPass(W1, W2, train_data)
    # out_hidden = 2 x 4
    # out_final = 2 x 2
    
    # backpropagation starts
    # error at the output node[s]
    temp_error = ((true_input * np.log(out_final)) + ((1 - true_input) * np.log(1 - out_final))) # temp_error = 2 x 2
    error = - 1 * np.sum(temp_error)/size_train_data# error = scalar
        
    # derivative of error function
    log_loss = out_final - true_input # log_loss = 2 X 2
    
    derivative_of_error_lj = np.dot(log_loss, out_hidden) # derivative_of_error_lj = 2 x 4
    temp_product = np.dot(log_loss.transpose(), W2) # temp_product = 2 x 4
    pairwise = (1 - out_hidden) * out_hidden * temp_product # pairwise = 2 x 4
    derivative_of_error_jp = np.dot(pairwise.transpose(), train_data)
    derivative_of_error_jp = np.delete(derivative_of_error_jp, n_hidden, 0) # derivative_of_error_jp = 3 x 6

    # performing regularization on the error to avoid overfitting
    squared_sum = np.sum(np.square(W1)) + np.sum(np.square(W2)) # squared_sum = scalar
    obj_val = error + ((lambdaval * squared_sum)/(2 * size_train_data)) # obj_val = scalar

    # performing regularization on the gradient
    # updating weights
    grad_W1 = (derivative_of_error_jp + (lambdaval * W1))/size_train_data # grad_W1 = 3 x 6
    grad_W2 = (derivative_of_error_lj + (lambdaval * W2))/size_train_data # grad_W1 = 2 x 4
    
    # Make sure you reshape the gradient matrices to a 1D array. for instance if
    # your gradient matrices are grad_W1 and grad_W2
    # you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_W1.flatten(), grad_W2.flatten()),0) # obj_grad = 26,
    
    # obj_val = 0
    # obj_grad = np.zeros (params.shape)

    return obj_val, obj_grad

def forwardPass(W1, W2, data):
    # this function is defined separately because you perform forward pass
    # for both nnObjFunction and nnPredict
    # It uses sigmoid activation
    
    # Forward pass at hidden layer
    aj = np.dot(W1, data.transpose()) # aj = 3 x 2
    out_hidden = sigmoid(aj) # out_hidden = 3 x 2
    
    # Adding bias to the data
    out_hidden = out_hidden.transpose()
    hidden_data = np.insert(out_hidden, out_hidden.shape[1], 1, axis = 1) # hidden_data = 2 x 4

    # Forward pass at output layer should be performed at w2 and hidden output
    bj = np.dot(W2, hidden_data.transpose()) # bj = 2 x 2
    out_final = sigmoid(bj) # out_final = 2 x 2
    
    return hidden_data, out_final


def nnPredict(W1, W2, data):
    '''
    % nnPredict predicts the label of data given the parameter W1, W2 of Neural
    % Network.
    % Input:
    % W1: matrix of weights for hidden layer units
    % W2: matrix of weights for output layer units
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image
    % Output:
    % label: a column vector of predicted labels
    '''
    #labels = np.zeros((data.shape[0],))
    # Your code here
    data = np.insert(data, data.shape[1], 1, axis = 1)
    out_hidden, out_final = forwardPass(W1, W2, data)
    labels = np.argmax(out_final,axis=0)
    
    return labels