import os
import pickle
from concurrent.futures import ProcessPoolExecutor as Executor

import numpy as np


def load(X=True, k=0, train=True, embed=False, numeric=True):
    """
    :param X: whether to load sequences or labels.
    :param k: index of the data set
    :param train: whether to load train or test data
    :param embed: whether to load the raw sequences or the embedding
    :param numeric: whether to load sequences of letters or integers
    :return: the corresponding data array.
    """
    data = np.loadtxt(
        'data/{}{}{}{}.csv'.format('X' if X else 'Y', 'tr' if train else 'te',
                                   k, '_mat100' if embed else ''),
        dtype=(float if embed else (str if X else bool)),
        skiprows=(0 if embed else 1),
        usecols=(None if embed else 1),
        delimiter=(' ' if embed else ','))
    if X and not embed and numeric:
        letters = np.array(list('ATCG'))
        data = np.array([list(s) for s in data])
        array = np.zeros(data.shape)
        for i, l in enumerate(letters):
            array[data == l] = i
        data = array
    return data


n_datasets = 3
train_Ys = [load(X=False, k=k) for k in range(n_datasets)]


def shuffle(*arrays):
    """
     Generates a random permutation and shuffle several arrays using it.
    """
    indices = np.arange(arrays[0].shape[0])
    np.random.shuffle(indices)
    return [a[indices] for a in arrays]


def k_folds_indices(n, k):
    """
     Return k pairs (train_inds, valid_inds) of arrays containing the training and validation indices for each split.
    """
    assert (n % k == 0)
    m = n // k
    indices = np.random.permutation(n)

    folds = []
    for i in range(k):
        folds.append((np.concatenate((indices[:i * m], indices[(i + 1) * m:])),
                      indices[i * m:(i + 1) * m]))
    return folds


def precomputed_kernels(kernel, name, numeric=True, max_workers=6, **params):
    """
    :param kernel: a function k(X, Y) that computes the kernels
    :param name: a unique name to represent the kernel
    :param numeric whether to load that data as numbers or strings
    :param max_workers parallel workers
    :param params kernel parameters
    :return a (train_Ks, test_Ks) tuple with kernels for all train and test datasets

    N.B. Stores the computed kernel to the disk, to reduce future computations.
    """
    kernels_dir = 'kernels'
    if not os.path.isdir(kernels_dir):
        os.mkdir(kernels_dir)
    file_name = '{}/{}'.format(kernels_dir, name)

    if os.path.exists(file_name):
        with open(file_name, 'rb') as file:
            kernels = pickle.load(file)
    else:
        train_Xs = [load(k=k, numeric=numeric) for k in range(n_datasets)]
        test_Xs = [load(k=k, train=False, numeric=numeric) for k in range(n_datasets)]

        with Executor(max_workers=max_workers) as executor:
            train_futures = [executor.submit(kernel, train_X, **params) for train_X in train_Xs]
            test_futures = [executor.submit(kernel, test_X, train_X, **params) for (test_X, train_X) in zip(test_Xs, train_Xs)]
            train_Ks = [future.result() for future in train_futures]
            test_Ks = [future.result() for future in test_futures]
        kernels = train_Ks, test_Ks
        with open(file_name, 'wb') as file:
            pickle.dump(kernels, file)

    return kernels


def transform_kernels(kernels, transform, **params):
    """
    :param kernels: a list of (train_Ks, test_Ks) tuples
    :param transform: a function that maps len(kernels) kernels to one kernel
    :return a (train_Ks, test_Ks) tuple with kernels for all train and test datasets
    """
    train, test = zip(*kernels)

    return (
        [transform(i, *ks, **params) for i, ks in enumerate(zip(*train))],
        [transform(i, *ks, **params) for i, ks in enumerate(zip(*test))],
    )


def save_predictions(predictions, file):
    """
    :param predictions: a list of length n_datasets with predictions on the test set of each dataset
    :param file: the name of the file to save predictions to.
    """
    predictions_dir = 'predictions'
    if not os.path.isdir(predictions_dir):
        os.mkdir(predictions_dir)
    predictions = np.concatenate(predictions)
    np.savetxt('{}/{}.csv'.format(predictions_dir, file),
               np.stack([np.arange(len(predictions)), predictions], axis=1),
               header='Id,Bound', comments='', fmt='%d', delimiter=',')