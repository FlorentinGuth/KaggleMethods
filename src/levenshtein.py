import Levenshtein
from tqdm import tqdm_notebook
import numpy as np
from concurrent.futures import ProcessPoolExecutor as Executor


def _compute_distance_row(args):
    x, Y = args
    row = np.zeros(len(Y))
    for j, y in enumerate(Y):
        row[j] = Levenshtein.distance(x, y)
    return row


def levenshtein_distance(X, Y=None, tqdm=False):
    if Y is None:
        Y = X
    d = np.zeros((len(X), len(Y)))
    with Executor(max_workers=12) as executor:
        iterator = enumerate(executor.map(_compute_distance_row, zip(X, [Y] * len(X))))
        if tqdm:
            iterator = tqdm_notebook(iterator, total=len(X))
        for i, row in iterator:
            d[i] = row
    return d
