###############################################################################
#
# AUTHOR(S): Samantha Muellner
#            Josh Kruse
# DESCRIPTION: program that will demonstrate regularization and overfitting
#           using Keras/Tensorflow
# VERSION: 1.0.0v
#
###############################################################################

import numpy as np
import csv, math
from math import sqrt
from matplotlib import pyplot as plt
import random

from sklearn.preprocessing import scale
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from keras.layers.core import Flatten, Dropout, Lambda
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras import optimizers
import tensorflow as tf


# global variables
MAX_EPOCHS = 100
DATA_FILE = "spam.data"


# Function: split matrix
# INPUT ARGS:
#   X_mat : matrix to be split
#   y_vec : corresponding vector to X_mat
# Return: train, validation, test
def split_matrix(X_mat, y_vec, size):
    # split data 80% train by 20% validation
    X_train, X_validation = np.split( X_mat, [int(size * len(X_mat))])
    y_train, y_validation = np.split( y_vec, [int(size * len(y_vec))])

    return (X_train, X_validation, y_train, y_validation)


# Function: convert_data_to_matrix
# INPUT ARGS:
#   file_name : the csv file that we will be pulling our matrix data from
# Return: data_matrix_full
def convert_data_to_matrix(file_name):
    with open(file_name, 'r') as data_file:
        spam_file = list(csv.reader(data_file, delimiter = " "))

    data_matrix_full = np.array(spam_file[0:], dtype=np.float)
    return data_matrix_full

# Function: sigmoid
# INPUT ARGS:
#   x : value to be sigmoidified
# Return: sigmoidified x
def sigmoid(x) :
    x = 1 / (1 + np.exp(-x))
    return x


# function that will create our NN model given the amount of units passed in
def create_model(units) :
    sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
    
    model = Sequential()
    
    model.add(Dense(units=units, activation='sigmoid', use_bias=False))
    model.add(Dense(1, activation="sigmoid", use_bias=False))
    
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    return model


# function to plot our loss
def plot_loss( res, vec ) :

    best = [ 0, 0, 0 ]
    
    for index, item in enumerate(res):
        plt.plot(item.history['loss'], label=str(vec[index]) + " train")
        min_index = np.argmin(item.history['loss'])
        plt.plot(min_index, item.history['loss'][min_index], "go")

        plt.plot(item.history['val_loss'], '--', label=str(vec[index]) + " val")
        res_best = np.argmin(item.history['val_loss'])
        res_loss = np.min(item.history['val_loss'])
        plt.plot(res_best, item.history['val_loss'][res_best], "go")

        if res_loss > best[2]:
            best[0] = index
            best[1] = res_best
            best[2] = res_loss

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.show()

    return best


# Function: main
def main():
    print("starting")
    # use spam data set

    data_matrix_full = convert_data_to_matrix(DATA_FILE)
    np.random.seed( 0 )
    np.random.shuffle(data_matrix_full)

    # get necessary variables
    # shape yields tuple : (row, col)
    col_length = data_matrix_full.shape[1]

    X_Mat = np.delete(data_matrix_full, col_length - 1, 1)
    y_vec = data_matrix_full[:,57]

    X_sc = scale(X_Mat)

    num_folds = 5
    multiplier_of_num_folds = int(X_Mat.shape[0]/num_folds)

    is_train = np.array(list(np.arange(1,
                                        num_folds + 1))
                                        * multiplier_of_num_folds)

    # make sure that test_fold_vec is the same size as X_Mat.shape[0]
    while is_train.shape[0] != X_sc.shape[0]:
        is_train = np.append(is_train, random.randint(1, num_folds))

    test_fold = 1
    
    # (10 points) Divide the data into 80% train, 20% test observations (out of all 
    # observations in the whole data set).
    is_train = np.random.choice( [True, False], X_sc.shape[0], p=[.8, .2] )
    # (10 points) Next divide the train data into 50% subtrain, 50% validation. 
    # e.g. as described here 
    # https://www.tensorflow.org/tutorials/keras/overfit_and_underfit#the_higgs_dataset
    subtrain_size = np.sum( is_train == test_fold )
    is_subtrain = np.random.choice( [True, False], subtrain_size, p=[.5, .5] )

    X_train = np.delete( X_sc, np.argwhere( is_subtrain != True ), 0)
    y_train = np.delete( y_vec, np.argwhere( is_subtrain != True ), 0)
    X_validation = np.delete( X_sc, np.argwhere( is_subtrain != False ), 0)
    y_validation = np.delete( y_vec, np.argwhere( is_subtrain != False ), 0)
    X_test = np.delete( X_sc, np.argwhere( is_train != False ), 0 )
    y_test = np.delete( y_vec, np.argwhere( is_train != False ), 0 )

    # (10 points) Define a for loop over regularization parameter values, and 
    # fit a neural network for each.
    hidden_units_vec = 2**np.arange(10)
    print(hidden_units_vec)

    units_matrix_list = []
    
    # loop over the different hidden units in hidden_units_vec
    for hidden_units_i in hidden_units_vec:
        # initialize keras model
        model = create_model(hidden_units_i) 

        # train on x-train, y-train
        # save results to data table (split_matrix_list) for further analysis
        units_matrix_list.append(model.fit( x = X_train,
                                y = y_train,
                                epochs = MAX_EPOCHS,
                                validation_data=(X_validation, y_validation),
                                verbose=2))

    print(units_matrix_list)
        

    # (20 points) On the same plot, show the logistic loss as a function of the 
    # regularization parameter (use a different color for each set, e.g. 
    # subtrain=solid, validation=dashed). Draw a point to emphasize the minimum 
    # of each validation loss curve. As the strength of regularization decreases, 
    # the train loss should always decrease, whereas the validation loss should 
    # decrease up to a certain point, and then start increasing (overfitting).
    # (10 points) Define a variable called best_parameter_value which is the 
    # regularization parameter value which minimizes the validation loss.
    best_tuple = plot_loss(units_matrix_list, hidden_units_vec)

    best_parameter_value = hidden_units_vec[best_tuple[0]]
    best_epoch_value = best_tuple[1]

    print("The best parameter value is ", best_parameter_value)

    # (10 points) Re-train the network on the entire train set (not just the 
    # subtrain set), using the corresponding value of best_parameter_value.
    final_model = create_model(best_parameter_value)
    # add output layer
    final_model.add(Dense(1, activation = "sigmoid", use_bias = False))

    result = final_model.fit( x = X_sc,
                        y = y_vec,
                        epochs = best_epoch_value,
                        verbose=2)

    # (10 points) Finally use the learned model to make predictions on the test 
    # set. What is the prediction accuracy? (percent correctly predicted labels 
    # in the test set) What is the prediction accuracy of the baseline model which 
    # predicts the most frequent class in the train labels?

    print("Prediction accuracy (correctly labeled) for the best parameter value is :", final_model.evaluate(X_test,y_test)[1])

    

    
main()

