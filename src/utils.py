import numpy as np
import pickle
import os
import svm
from concurrent.futures import ProcessPoolExecutor as Executor


def load(X=True, k=0, train=True, embed=False, numeric=True):
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


def evaluate(classifier, K, Y, folds=5):
    """
    :param classifier: classifier to evaluate
    :param K: precomputed kernel matrix of shape (n_samples, n_samples)
    :param Y: training labels of shape (n_samples, )
    :param folds: number of folds to use

    Evaluates the classifier using cross-validation.
    Returns the mean and std of the validation and train scores.:
        (valid_scores_mean, valid_scores_std, train_scores_mean, train_scores_std)
    """

    N = len(K)
    valid_scores = []
    train_scores = []
    for train_inds, valid_inds in k_folds_indices(N, folds):
        classifier.fit(K[np.ix_(train_inds, train_inds)], Y[train_inds])

        valid_scores.append((classifier.predict(K[np.ix_(
            valid_inds, train_inds)]) == Y[valid_inds]).mean())
        train_scores.append((classifier.predict(K[np.ix_(
            train_inds, train_inds)]) == Y[train_inds]).mean())

    valid_scores = np.array(valid_scores)
    train_scores = np.array(train_scores)

    return valid_scores.mean(), valid_scores.std(), train_scores.mean(), train_scores.std()


def global_evaluate(classifier, Ks, Ys, Cs, folds=5, **params):
    """
    :param classifier: classifier to evaluate
    :param Ks: list precomputed kernel matrices of shape list(n_samples_i, n_samples_i)
    :param Ys: training labels of shape list(n_samples_i,)
    :param Cs: list of parameters C
    :param folds: number of folds to use
    :param params: additional parameters to instantiate the classifier

    Evaluates a classifier on several data sets, and averages the results.
    """
    return np.mean(
        np.array([
            evaluate(classifier(C=C, **params), K, Y, folds)
            for (K, Y, C) in zip(Ks, Ys, Cs)
        ]),
        axis=0)


def precomputed_kernels(kernel, name, numeric=True, **params):
    """
    :param kernel: a function k(X, Y) that computes the kernels
    :param name: a unique name to represent the kernel
    :param numeric whether to load that data as numbers or strings
    :param params kernel parameters
    :return a (train_Ks, test_Ks) tuple with kernels for all train and test datasets
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

        with Executor(max_workers=6) as executor:
            train_futures = [executor.submit(kernel, train_X, train_X, **params) for train_X in train_Xs]
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


def grid_search(model, params, K, Y, folds=5, no_perfect=False):
    """
    :param model: a function that instantiates a model given parameters from params
    :param params: list of parameters to try
    :param K the kernel
    :param Y the labels
    :param folds number of folds for validation
    :param no_perfect should be set to True to remove parameters that lead to 100% training accuracy
    :return selected parameters and associated performance
     Grid search on params using k-fold cross validation.
    """
    results = []
    for p in params:
        results.append(np.array(evaluate(model(**p), K, Y, folds=folds)))

    results = np.array(results)
    if no_perfect:
        non_perfect = results[:, 2] < 1
        p = params[np.argmax(results[non_perfect][:, 0] - results[non_perfect][:, 1])]
    else:
        p = params[np.argmax(results[:, 0] - results[:, 1])]
    return p, np.array(evaluate(model(**p), K, Y, folds=20))


def final_train(model, p, K_train, Y_train, K_test):
    """
    :param model: A model constructor
    :param p: Model parameters
    :param K_train: Training kernel
    :param Y_train: Training labels
    :param K_test: Test kernel
    :return: Predictions of the trained model on the test data
    """
    m = model(**p)
    m.fit(K_train, Y_train)
    return m.predict(K_test)


def save_predictions(predictions, file):
    """
    :param predictions: A list of length n_datasets with predictions on the test set of each dataset
    :param file: the name of the file to save predictions to
    """
    predictions_dir = 'predictions'
    if not os.path.isdir(predictions_dir):
        os.mkdir(predictions_dir)
    predictions = np.concatenate(predictions)
    np.savetxt('{}/{}.csv'.format(predictions_dir, file),
               np.stack([np.arange(len(predictions)), predictions], axis=1),
               header='Id,Bound', comments='', fmt='%d', delimiter=',')


def svm_kernels(kernels, prediction_file=None):
    train_Ks, test_Ks = kernels

    model = svm.SVC
    params = [dict(kernel='precomputed', C=C) for C in 2. ** np.arange(0, 10)]
    total_perf = np.zeros(4)
    predictions = []
    for K, Y, K_test in zip(train_Ks, train_Ys, test_Ks):
        p, performance = grid_search(model, params, K, Y)
        total_perf += performance
        percentages = tuple(100 * performance)
        print('dataset: Validation {:.2f} ± {:.2f}\t Train {:.2f} ± {:.2f}\t C={:.0e}'.format(*percentages, p['C']))

        if prediction_file is not None:
            predictions.append(final_train(model, p, K, Y, K_test))

    total_percentages = 100 * total_perf / 3
    print('total:   Validation {:.2f} ± {:.2f}\t Train {:.2f} ± {:.2f}\t'.format(*total_percentages))

    if prediction_file is not None:
        save_predictions(predictions, prediction_file)
