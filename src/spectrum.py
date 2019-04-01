import numpy as np
from scipy import sparse
import scipy.sparse.linalg


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
    strides = a.strides + (a.strides[-1],)

    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def sparse_prod(a, b, n, m):
    """
    Computes K = A.T.dot(B) from sparse representations a, b of A and B.
    :param a: indices and values from matrix a
    :param b: indices and values from matrix b
    :param n: size of second axis of input matrix a
    :param m: size of second axis of input matrix b
    :return: the matrix K = a.T.dot(b)
    """
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


def sparse_norm(a, n):
    """
    Computes the norm of each column of the matrix A from sparse representation a of A.
    :param a: indices and values of matrix a
    :param n: size of second axis of matrix a
    :return: norm of the columns of A.
    """
    inds_a, counts_a = a

    norms2 = np.zeros(n)

    for (i, x), count in zip(inds_a, counts_a):
        norms2[int(x)] += count * count

    return np.sqrt(norms2)


def k_grams(X, k=2, m=4, ret_inds=False):
    """
    :param X: matrix of size (N, L) containing N sequences of L integers
    :param k: size of the k-grams to extract
    :param m: size of the alphabet
    :param ret_inds: to return sparse representation of the matrix
    :return: sparse matrix of size (N, m ** k) containing counts of k-grams.
    """

    N, L = X.shape

    X = X.astype(np.int32)

    dtype = np.uint64 if not ret_inds else object
    encoding = m ** np.arange(k, dtype=dtype)
    assert(encoding.dtype == dtype)

    k_grams_indices = rolling_window(X, window=k).dot(encoding)

    xs = np.repeat(np.arange(N), repeats=k_grams_indices.shape[1])
    ys = k_grams_indices.reshape(-1)

    if ret_inds:
        zs = np.empty(len(ys), dtype=object)
        for i, (x, y) in enumerate(zip(xs, ys)):
            zs[i] = (y, x)
        inds, counts = np.unique(zs, return_counts=True)
        return inds, counts

    inds = np.stack((xs, ys), axis=1)
    inds, counts = np.unique(inds, axis=0, return_counts=True)

    return sparse.csr_matrix((counts, (inds[:, 0], inds[:, 1])),
                             shape=(N, m ** k),
                             dtype=int)


def k_spectrum(X, Y=None, k=2, m=4):
    """
     Computes the k-spectrum kernel between the sequences from X and Y.
     If Y is None, computes the k-spectrum between sequences from X.

     NB: 'normalizes' the kernel.
     NB: for k > 12, it uses `k_spectrum_extreme`.
    """
    if k > 13:
        return k_spectrum_extreme(X, Y, k, m)

    k_grams_X = k_grams(X, k=k, m=m)
    k_grams_Y = k_grams_X if Y is None else k_grams(Y, k=k, m=m)

    norms_X = sparse.linalg.norm(k_grams_X, axis=1)
    norms_Y = sparse.linalg.norm(k_grams_Y, axis=1)

    K = k_grams_X.dot(k_grams_Y.T).toarray() / np.outer(norms_X, norms_Y)

    return K


def k_spectrum_extreme(X, Y=None, k=2, m=4):
    """
     Computes the k-spectrum kernel between X and Y, just as `k_spectrum`,
     using only sparse representations of the matrices.
    """
    k_grams_inds_X = k_grams(X, k=k, m=m, ret_inds=True)
    k_grams_inds_Y = k_grams_inds_X if Y is None else k_grams(
        Y, k=k, m=m, ret_inds=True)

    K = sparse_prod(k_grams_inds_X, k_grams_inds_Y, len(X),
                    len(X) if Y is None else len(Y))

    norms_X = sparse_norm(k_grams_inds_X, len(X))
    norms_Y = norms_X if Y is None else sparse_norm(k_grams_inds_Y, len(Y))

    K /= np.outer(norms_X, norms_Y)

    return K


def mismatch_permutations(k):
    """
     Computes an iterator over all possible modifications of the k-grams with one mismatch.
     Each returned element is an array P such that k-gram P[i] has exactly one mismatch with k-gram i.
    """
    for pos in range(k):
        for shift in range(3):
            ind = np.arange(1 << (2 * k))
            cur = (ind >> (2 * pos)) % 4
            new = (cur + 1 + shift) % 4
            yield ind + ((new - cur) << (2 * pos))


def k_spectrum_mismatch(X, Y=None, k=2, decay=1):
    """
     Computes the (k, 1)-mismatch kernel between the sequences from X and Y.
     If Y is None, computes the kernel between sequences from X.
    """
    m = 4
    k_grams_X = k_grams(X, k=k, m=m)
    k_grams_Y = k_grams_X if Y is None else k_grams(Y, k=k, m=m)

    k_grams_X_mis = k_grams_X.copy()
    k_grams_Y_mis = k_grams_Y.copy()

    for perm in mismatch_permutations(k):
        k_grams_X_mis += decay * k_grams_X[:, perm]
        k_grams_Y_mis += decay * k_grams_Y[:, perm]

    norms_X = sparse.linalg.norm(k_grams_X_mis, axis=1)
    norms_Y = sparse.linalg.norm(k_grams_Y_mis, axis=1)

    K = k_grams_X_mis.dot(k_grams_Y_mis.T).toarray() / np.outer(norms_X, norms_Y)
    return K

