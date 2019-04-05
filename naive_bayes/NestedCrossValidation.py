# ANSWER HERE: Suggested code structure in comments below

from sklearn import datasets
import operator
import numpy as np
from math import *
from decimal import Decimal

# LOAD IRIS DATA SET HERE
iris = datasets.load_iris()
# Assigning data. Features array denoted with capital X and target array denoted with small y
X = iris.data
y = iris.target

def split_train_test_1(data, test_size):
    # TASK: use numpy.random.permutation to obtain a randomly ordered vector of indices from 1...length of our data
    # use X.shape[...] to obtain the length of our data - replace ... with correct dimension
    indices = np.random.permutation(len(data))

    # TASK: determine number of data:  should be test_size times the total number of data (see above)
    test_len = np.floor(test_size * len(data)).astype(int)

    # TASK: gather the data indices to be used for training in the idxTrain variable
    # should be equal to the data points from 0 to T-testlen, where T is the total number of data.
    # E.g., if testLen is 2 and T=10, then we take elements [0,1,...,7] since T-2=8
    indices_train = indices[:-test_len]

    # TASK: gather the data indices to be used for testing in the idxTest variable
    # Should contain indices from -testLen to T
    indices_test = indices[-test_len:]

    # make sure that: no indices are used for both training and testing,
    # and that the total number of data in the training and testing set equal to the total number of data
    # assert not L makes sure list L is empty: for more on assert check the python documentation
    assert not np.intersect1d(indices_test, indices_train)
    assert (len(indices_train) + len(indices_test) == X.shape[0])

    # create the training and testing datasets based on the indices defined above
    x_train = X[indices_train]
    x_test = X[indices_test]
    y_train = y[indices_train]
    y_test = y[indices_test]

    # to return multiple values from a function in python, we need to use tuples
    # more on tuples here: https://www.tutorialspoint.com/python/python_tuples.htm
    return indices_train, indices_test, x_train, x_test, y_train, y_test

def euclidean_distance(in1, in2, length):
    distance = 0
    for x in range(length):
        distance += pow((in1[x] - in2[x]), 2)
    return np.math.sqrt(distance)  # euclidean distance between in1 and in2



def get_neighbors(training_set, test_instance, train_set_actual):
    distances_ = []
    l = len(test_instance)
    for x in range(len(training_set)):
        x__ = train_set_actual[x]
        dist = euclidean_distance(test_instance, x__, l)  # comment / uncomment to use euclidean
        # dist = minkowski_distance(test_instance, x_, 3) # comment / uncomment to use minkowski distance
        distances_.append((training_set[x], dist))
    distances_.sort(key=operator.itemgetter(1))
    neighbors_ = []
    for x in range(k):
        neighbors_.append(distances_[x][0])
        # return indices of n-nearest neighbours in
        # training data and distances between test ant train data points
        #print('neighb ', neighbors_)
        #print('dist ', distances_)
    return neighbors_, distances_

# here is some sample code for evaluating the kNN classifier you just built
# NOTE: this is just a suggested way to do this - you can do it in another way if you want

# use function to split data to training and testing - use 20% of the dataset for testing, 80% for training
idxTrain, idxTest, X_train, X_test, y_train, y_test = split_train_test_1(X, 0.2)




correct = 0
incorrect = 0
predictions = []
for i in idxTest:  # for all test points
    # knn classifier
    x_ = X[i]  # test point x_
    y_ = y[i]  # true label for y_
    k = 10  # Number of neighbours
    # to get nearest neighbours, passing training indices, test instance and training set
    neighbors, distances = get_neighbors(idxTrain, x_, X_train)
print('neighb ', neighbors)
print('dist ', distances)
    # assign label to predicted x_

    # add to the prediction set



