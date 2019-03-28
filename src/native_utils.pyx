import numpy as np
cimport numpy as np
import cython

# noinspection PyUnresolvedReferences
@cython.boundscheck(False)
@cython.wraparound(False)
def levenshtein_one_vs_many(np.ndarray[np.int_t, ndim=1] a, np.ndarray[np.int_t, ndim=2] bs,
                            np.ndarray[np.float32_t, ndim=1] weights):
    cdef np.ndarray[np.int_t, ndim=2] sub_op = np.zeros((4, 4), dtype=np.int)
    sub_op[0] = [-1, 4, 5, 6]
    sub_op[1] = [4, -1, 7, 8]
    sub_op[2] = [5, 7, -1, 9]
    sub_op[3] = [6, 8, 9, -1]

    cdef int l_a = a.shape[0]
    cdef int l_b = bs.shape[1]
    cdef np.ndarray[np.float32_t, ndim=1] prev_row = np.zeros(l_b + 1, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] cur_row  = np.zeros(l_b + 1, dtype=np.float32)

    cdef np.ndarray[np.float32_t, ndim=1] dists = np.zeros(bs.shape[0], dtype=np.float32)

    cdef int i, j, k
    for k in range(bs.shape[0]):
        prev_row[0] = 0
        for j in range(l_b):
            prev_row[j + 1] = prev_row[j] + weights[bs[k, j]]
        for i in range(l_a):
            cur_row[0] = prev_row[0] + weights[a[i]]

            for j in range(l_b):
                cur_row[j + 1] = min(
                    prev_row[j + 1] + weights[a[i]],
                    cur_row[j] + weights[bs[k, j]],
                    prev_row[j] if a[i] == bs[k, j] else prev_row[j] + weights[sub_op[a[i], bs[k, j]]]
                )
            cur_row, prev_row = prev_row, cur_row

        dists[k] = prev_row[l_b]
    return dists