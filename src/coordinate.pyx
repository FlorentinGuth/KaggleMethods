import numpy as np
cimport numpy as np
import cython

# noinspection PyUnresolvedReferences
@cython.boundscheck(False)
@cython.wraparound(False)
def coordinate_descent(np.ndarray[np.float32_t, ndim=2] k, np.ndarray[np.int_t, ndim=1] y,
                       float C=1., int n_iter=1000, float tol=1e-3, str loss='hinge'):
    cdef int n = k.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2] k_C = k
    cdef np.ndarray[np.float32_t, ndim=1] alpha = np.zeros(n, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] Y = (2 * y - 1).astype(np.float32)
    cdef np.ndarray[np.int_t, ndim=1] perm = np.empty(n, dtype=np.int)
    cdef np.float32_t a_max

    if loss == 'hinge':
        a_max = C
    else:
        k_C += np.eye(n, dtype=np.float32)/(2*C)
        a_max = np.inf

    cdef np.float32_t bound = 0.
    cdef np.float32_t prev_alpha = 0
    cdef np.float32_t y_i = 0

    for j in range(n_iter):
        perm = np.random.permutation(n)
        bound = 0
        for i in perm:
            prev_alpha = alpha[i]
            y_i = Y[i]
            alpha[i] = y_i * min(max(0, y_i * (prev_alpha + (y_i - k_C[i].dot(alpha)) / k_C[i, i])), a_max)
            bound = max(bound, np.abs(alpha[i] - prev_alpha))
        if bound < tol:
            break
    return alpha
