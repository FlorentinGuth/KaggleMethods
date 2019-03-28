import native_utils
from tqdm import tqdm as tqdm_notebook
import numpy as np
from concurrent.futures import ProcessPoolExecutor as Executor
import multiprocessing


def _compute_distance_row(weights, a, bs):
    return native_utils.levenshtein_one_vs_many(a, bs, weights)


def levenshtein_distance(weights, X, Y=None, tqdm=False):
    if Y is None:
        Y = X
        Ys = [Y[i:] for i in range(len(X))]
    else:
        Ys = [Y] * len(X)
    d = np.zeros((len(X), len(Y)), dtype=np.float32)
    with Executor(max_workers=multiprocessing.cpu_count()) as executor:
        iterator = enumerate(executor.map(_compute_distance_row, [weights] * len(X), X, Ys))
        if tqdm:
            iterator = tqdm_notebook(iterator, total=len(X))
        for i, row in iterator:
            if len(row) < len(X):
                d[i, i:] = row
                d[i, :i] = d[:i, i]
            else:
                d[i] = row
    return d


def main():
    import utils
    weights = np.ones(10, dtype=np.float32)
    weights[:4] = 2
    dists = levenshtein_distance(weights, utils.load(k=0).astype(np.int)[:2000], tqdm=True)
    print(dists[dists != 0].max(), dists[dists != 0].mean(), dists[dists != 0].min())


if __name__ == '__main__':
    main()
