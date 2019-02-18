import numpy as np


def load(X=True, k=0, train=True, embed=False):
    data = np.loadtxt(
        'data/{}{}{}{}.csv'.format('X' if X else 'Y', 'tr' if train else 'te',
                                   k, '_mat100' if embed else ''),
        dtype=(float if embed else (str if X else bool)),
        skiprows=(0 if embed else 1),
        usecols=(None if embed else 1),
        delimiter=(' ' if embed else ','))
    if X and not embed:
        letters = np.array(list('ATCG'))
        data = np.array([list(s) for s in data])
        numeric = np.zeros(data.shape)
        for i, l in enumerate(letters):
            numeric[data == l] = i
        data = numeric
    return data


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

    return valid_scores.mean(), valid_scores.std(), train_scores.mean(
    ), train_scores.std()


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
