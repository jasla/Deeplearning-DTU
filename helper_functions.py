import numpy as np
from sklearn.utils import shuffle


def load_data():
    # To speed up training we'll only work on a subset of the data containing only the numbers 0, 1.
    data = np.load('mnist.npz')
    # Possible classes
    classes = list(range(10))
    # Set classes used, starting from 0.
    # included_classes = [0, 1, 4, 9]
    # included_classes = [0, 1, 4, 9, 5, 8]
    included_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    idxs_train = []
    idxs_valid = []
    idxs_test = []
    num_classes = 0
    for c in included_classes:
        if c in classes:
            num_classes += 1
            idxs_train += np.where(data['y_train'] == c)[0].tolist()
            idxs_valid += np.where(data['y_valid'] == c)[0].tolist()
            idxs_test += np.where(data['y_test'] == c)[0].tolist()
    print("Number of classes included:", num_classes)
    # x_train = data['X_train'][idxs_train].astype('float32')
    x_train = data['X_train'].astype('float32')
    # targets_train = data['y_train'][idxs_train].astype('int32')
    targets_train = data['y_train'].astype('int32')
    x_train, targets_train = shuffle(x_train, targets_train, random_state=1234)
    # x_valid = data['X_valid'][idxs_valid].astype('float32')
    # targets_valid = data['y_valid'][idxs_valid].astype('int32')
    x_valid = data['X_valid'].astype('float32')
    targets_valid = data['y_valid'].astype('int32')
    # x_test = data['X_test'][idxs_test].astype('float32')
    # targets_test = data['y_test'][idxs_test].astype('int32')
    x_test = data['X_test'].astype('float32')
    targets_test = data['y_test'].astype('int32')
    print("training set dim(%i, %i)." % x_train.shape)
    print("validation set dim(%i, %i)." % x_valid.shape)
    print("test set dim(%i, %i)." % x_test.shape)

    return included_classes, num_classes, c, x_train, targets_train, x_valid, targets_valid, x_test, targets_test