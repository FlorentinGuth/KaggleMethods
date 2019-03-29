import numpy as np
from scipy import sparse
import scipy.sparse.linalg
from tqdm import tqdm_notebook


def rolling_window(a, window):
    """
    Make an ndarray with a rolling window of the last dimension

    Parameters
    ----------
    a : array_like
       Array to add rolling window to
    window : int
       Size of rolling window

    Returns
    -------
    Array that is a view of the original array with a added dimension
    of size w.

    Examples
    --------
    >>> x=np.arange(10).reshape((2,5))
    >>> rolling_window(x, 3)
    array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
          [[5, 6, 7], [6, 7, 8], [7, 8, 9]]])

    Calculate rolling mean of last dimension:
    >>> np.mean(rolling_window(x, 3), -1)
    array([[ 1.,  2.,  3.],
          [ 6.,  7.,  8.]])

    """
    if window < 1:
        raise ValueError("`window` must be at least 1.")
    if window > a.shape[-1]:
        raise ValueError("`window` is too long.")

    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1], )

    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def sparse_prod(a, b, n, m):
    inds_a, counts_a = a
    inds_b, counts_b = b

    kernel = np.zeros((n, m))
    table_a = dict()

    last_i = -1
    xs = []
    counts = []

    for (i, x), count in zip(inds_a, counts_a):
        if i != last_i:
            table_a[last_i] = (xs, counts)

            xs = []
            counts = []
            last_i = i

        xs.append(int(x))
        counts.append(count)

    table_a[last_i] = (xs, counts)

    for (j, y), count in zip(inds_b, counts_b):
        if j not in table_a:
            continue
        xs, counts = table_a[j]

        kernel[xs, int(y)] += np.asarray(counts) * count

    return kernel


def k_grams(X, k=2, m=4, ret_inds=False):
    N, L = X.shape

    encoding = (m ** np.arange(k, dtype=np.uint64)).astype(np.uint64)

    k_grams_indices = rolling_window(X, window=k).dot(encoding)

    xs = np.repeat(np.arange(N), repeats=k_grams_indices.shape[1])
    ys = k_grams_indices.reshape(-1)

    inds = np.stack((xs, ys), axis=1)
    inds, counts = np.unique(inds, axis=0, return_counts=True)

    res = sparse.csr_matrix((counts, (inds[:, 0], inds[:, 1])),
                            shape=(N, m**k),
                            dtype=int)
    if ret_inds:
        return res, np.unique(
            np.stack((ys, xs), axis=1), axis=0, return_counts=True)
    return res


def k_spectrum(X, Y=None, k=2, m=4):
    """
     Computes the k-spectrum kernel between the sequences from X and Y.
     If Y is None, computes the k-spectrum between sequences from X.

     NB: 'normalizes' the kernel, but does not center it.
    """
    if k > 13:
        return k_spectrum_extreme(X, Y, k, m)

    k_grams_X = k_grams(X, k=k, m=m)
    k_grams_Y = k_grams_X if Y is None else k_grams(Y, k=k, m=m)

    # TODO: Normalization? Centering?
    norms_X = sparse.linalg.norm(k_grams_X, axis=1)
    norms_Y = sparse.linalg.norm(k_grams_Y, axis=1)

    K = k_grams_X.dot(k_grams_Y.T).toarray() / np.outer(norms_X, norms_Y)

    return K


def k_spectrum_extreme(X, Y=None, k=2, m=4):
    k_grams_X, k_grams_inds_X = k_grams(X, k=k, m=m, ret_inds=True)
    k_grams_Y, k_grams_inds_Y = (k_grams_X,
                                 k_grams_inds_X) if Y is None else k_grams(
                                     Y, k=k, m=m, ret_inds=True)

    K = sparse_prod(k_grams_inds_X, k_grams_inds_Y, len(X),
                    len(X) if Y is None else len(Y))

    # TODO: Normalization? Centering?
    norms_X = sparse.linalg.norm(k_grams_X, axis=1)
    norms_Y = sparse.linalg.norm(k_grams_Y, axis=1)

    K /= np.outer(norms_X, norms_Y)

    return K


def cum_spectrum(X, Y=None, k=5, tqdm=False):
    """
     Computes the sum of the spectrum kernels between X and Y, up to k.
    """
    shape = (len(X), len(X) if Y is None else len(Y))
    kernel = np.zeros(shape)

    for i in (tqdm_notebook(range(1, k + 1)) if tqdm else range(1, k + 1)):
        kernel += k_spectrum(X, Y=Y, k=i)

    return kernel / k


def k_spectra(X, Y=None, k=5, tqdm=False):
    """
     Computes the spectrum kernels between X and Y, up to k.
    """
    shape = (len(X), len(X) if Y is None else len(Y))
    kernel = np.zeros((k,) + shape)

    for i in (tqdm_notebook(range(1, k + 1)) if tqdm else range(1, k + 1)):
        kernel[i-1] = k_spectrum(X, Y=Y, k=i)

    return kernel


def mismatch_permutations(k):
    for pos in range(k):
        for shift in range(3):
            ind = np.arange(1 << (2 * k))
            cur = (ind >> (2 * pos)) % 4
            new = (cur + 1 + shift) % 4
            yield ind + ((new - cur) << (2 * pos))


def k_spectrum_mismatch(X, Y=None, k=2, decay=.5):
    m = 4
    k_grams_X = k_grams(X, k=k, m=m)
    k_grams_Y = k_grams_X if Y is None else k_grams(Y, k=k, m=m)

    for perm in mismatch_permutations(k):
        k_grams_X += decay * k_grams_X[:, perm]
        k_grams_Y += decay * k_grams_Y[:, perm]

    norms_X = sparse.linalg.norm(k_grams_X, axis=1)
    norms_Y = sparse.linalg.norm(k_grams_Y, axis=1)

    K = k_grams_X.dot(k_grams_Y.T).toarray() / np.outer(norms_X, norms_Y)
    return K


def cum_mismatch(X, Y=None, k=5, tqdm=False, decay=.1):
    """
     Computes the sum of the spectrum kernels between X and Y, up to k.
    """
    shape = (len(X), len(X) if Y is None else len(Y))
    kernel = np.zeros(shape)

    for i in (tqdm_notebook(range(1, k + 1)) if tqdm else range(1, k + 1)):
        if 3 <= i <= 6:
            kernel += k_spectrum_mismatch(X, Y=Y, k=i, decay=decay)
        else:
            kernel += k_spectrum(X, Y=Y, k=i)

    return kernel / k
