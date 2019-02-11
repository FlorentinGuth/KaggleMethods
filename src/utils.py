import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

def load(X=True, k=0, train=True, embed=False):
    data = np.loadtxt('data/{}{}{}{}.csv'.format(
        'X' if X else 'Y', 'tr' if train else 'te', k, '_mat100' if embed else ''),
                      dtype=(float if embed else (str if X else bool)),
                      skiprows=(0 if embed else 1),
                      usecols=(None if embed else 1),
                      delimiter=(' ' if embed else ',')
    )
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

def folds_split(k, *arrays):
    n = arrays[0].shape[0]
    assert n % k == 0
    arrays = shuffle(*arrays)
    fold_size = n // k
    for i in range(k):
        yield [
            (np.concatenate((a[:i * fold_size], a[(i+1) * fold_size:])), a[i * fold_size:(i+1) * fold_size])
            for a in arrays
        ]

def evaluate(classifier, X, Y, folds=5):
    scores = []
    for (X_t, X_v), (Y_t, Y_v) in folds_split(folds, X, Y):
        classifier.fit(X_t, Y_t)
        Y_p = classifier.predict(X_v)
        scores.append((Y_p == Y_v).mean())
    scores = np.array(scores)
    return scores.mean(), scores.std()
