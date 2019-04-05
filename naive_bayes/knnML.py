from sklearn import datasets
import numpy as np


# Load the IRIS dataset, as in the labs
mySeed=1234567

import matplotlib.pyplot as plt
import operator

iris = datasets.load_iris()
X=iris.data
y=iris.target



def train_test_split(X, y, test_size):
    # creates an array with shuffled indices of the dataset
    indices = np.random.permutation(len(X))
    # print(indices)

    # test_length is the size of the test sample
    test_length = int(np.floor(test_size * len(X)))
    # print(test_length)

    #dataset is plit into train/test
    train_indices = indices[:-test_length]
    # print(train_indices)
    test_indices = indices[-test_length:]
    # print(test_indices)

    # asserts no indices are used for both sets
    assert not np.intersect1d(train_indices, test_indices)
    # asserts the the lengths match
    assert (len(train_indices) + len(test_indices) == X.shape[0])

    # the new datasets are created using the indice sizes
    x_train = X[train_indices]
    x_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return x_train, x_test, y_train, y_test


def euclidian_distance(value_1, value_2, length):
    distance = 0
    for i in range(length):
        distance += pow(value_1[i] - value_2[i], 2)
    return np.math.sqrt(distance)

def manhattan_distance(value_1, value_2, length):
    distance = 0
    for i in range(length):
        distance += abs(value_1[i] - value_2[i])
    return distance


def mykNN(x_train, y_train, x_test, nn, dist):
    prediction = []
    for i in range(len(x_test)):
        distances = []
        x = x_test[i]
        x_size = len(x)
        for j in range(len(x_train)):
            if dist == 'Euclidian':
                distance = euclidian_distance(x_train[j], x, x_size)
                if distance != 0.0:
                    distances.append((j, distance))
            elif dist == 'Manhattan':
                distance = manhattan_distance(x_train[j], x, x_size)
                if distance != 0.0:
                    distances.append((j, distance))
        distances.sort(key=operator.itemgetter(1))
        neighbours = []
        for j in range(nn):
            neighbours.append(distances[j][0])
        #print('neighb ', neighbours)
        #print('dist ', distances)
        distances__ = {}
        count = 0
        for j in neighbours:
            label = y_train[j]
            dis = distances[count][1]
            count += 1
            for item, value in distances__.items():
                if item == label:
                    dis += value
                    break
            weight = 1 / (1+dis)
            distances__[label] = weight + dis
        m = max(distances__, key=lambda key_in: distances__[key_in])
        prediction.append(m)
    return np.asarray(prediction)

############################################################
#x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#y_prediction = mykNN(x_train, y_train, x_test, nn=1, dist='Manhattan')
#print(y_prediction)
#print(y_test)


def myNestedCrossVal(X, y, foldK, nns, dists, mySeed):
    np.random.seed(mySeed)
    accuracy_fold = []
    #X = X + np.random.normal(0, 0.4, X.shape)
    # TASK: use the function np.random.permutation to generate a list of shuffled indices from in the range (0,number of data)
    # (you did this already in a task above)
    indices = np.random.permutation(X.shape[0])
    # print(indices)

    # TASK: use the function array_split to split the indices to foldK different bins (here, 5)
    # uncomment line below
    bins = np.array_split(indices, foldK)
    # print(bins)

    # no need to worry about this, just checking that everything is OK
    assert (foldK == len(bins))

    # loop through folds
    for i in range(0, foldK):
        foldTrain = []  # list to save current indices for training
        foldTest = []  # list to save current indices for testing
        foldVal = []  # list to save current indices for validation

        # loop through all bins, take bin i for testing, the next bin for validation, and the rest for training
        for j in range(0, len(bins)):
            # insert code here
            index = 0
            # print(index)
            # foldTest.append(bins[i])
            foldTest = bins[i]
            if i < foldK - 1:
                index = i + 1
                foldVal = bins[index]
            else:
                index = i - (foldK - 1)
                foldVal = bins[index]
            if j != i and j != index:
                foldTrain.extend(bins[j])
        # print('** Train', len(foldTrain), foldTrain)
        # print('** Val', len(foldVal), foldVal)
        # print('** Test', len(foldTest), foldTest)


        # no need to worry about this, just checking that everything is OK
        assert not np.intersect1d(foldTest, foldVal)
        assert not np.intersect1d(foldTrain, foldTest)
        assert not np.intersect1d(foldTrain, foldVal)

        x_train = X[foldTrain]
        x_test = X[foldTest]
        x_val = X[foldVal]
        y_train = y[foldTrain]
        y_test = y[foldTest]
        y_val = y[foldVal]
        accuracy_score = 0.0
        final_accuracy_score = 0.0
        bestDistance = ''  # save the best distance metric here
        bestNN = -1  # save the best number of neighbours here
        bestAccuracy = -10  # save the best attained accuracy here (in terms of validation)

        # loop through all parameters (one for loop for distances, one for loop for nn)
        # train the classifier on current number of neighbours/distance
        # obtain results on validation set
        # save parameters if results are the best we had
        for k in range(0, len(dists)):
            for l in (nns):
                #knn = mykNN(x_train, y_train, x_val, nn=l, dist=dists[k])

                # knn=KNeighborsClassifier(n_neighbors=l, metric=dists[k])
                # knn.fit(x_train,y_train)
                y_pred = mykNN(x_train, y_train, x_val, nn=l, dist=dists[k])
                accuracy = 0.0
                for m in range (len(y_val)):
                    if y_val[m] == y_pred[m]:
                        accuracy += 1
                accuracy_score = accuracy/len(y_val)
                print('validation for number of neighbours = ', l, 'distance = ', dists[k], 'accuracy = ', accuracy_score * 100)
                if (accuracy_score * 100) > bestAccuracy:
                    bestAccuracy = (accuracy_score * 100)
                    bestNN = l
                    bestDistance = dists[k]

        print(l,'** End of validation for fold,', i, 'best Number of Neighbours:', bestNN, 'best Distance: ', bestDistance)

        # evaluate on test data:
        # extend your training set by including the validation set


        x_train = X[np.append(foldTrain, foldVal)]
        y_train = y[np.append(foldTrain, foldVal)]
        assert (len(x_train) + len(x_test) == X.shape[0])





        # train k-NN classifier on new training set and test on test set
        #knn = KNeighborsClassifier(n_neighbors=bestNN, metric=bestDistance)
        #knn.fit(x_train, y_train)
        final_pred = mykNN(x_train, y_train, x_test, nn=bestNN, dist=bestDistance)
        print(final_pred)
        print(y_test)
        accuracy = 0
        for n in range(len(final_pred)):
            if y_test[n] == final_pred[n]:
                accuracy += 1
        final_accuracy_score = accuracy / len(final_pred)
        accuracy_fold.append(final_accuracy_score*100)
        #accuracy_fold.append((final_accuracy_score * 100))
        # get performance on fold, save result in accuracy_fold array



        print('==== Final Cross-val on test on this fold with NN', bestNN, 'dist', bestDistance, ' accuracy ',
              final_accuracy_score * 100)

    return accuracy_fold;


# call your nested crossvalidation function:

accuracy_fold = myNestedCrossVal(X, y, 5, list(range(1, 11)), ['Euclidian', 'Manhattan'], mySeed)
print(accuracy_fold)