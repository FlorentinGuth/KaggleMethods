import numpy as np
cimport numpy as np
import cython

# noinspection PyUnresolvedReferences
@cython.boundscheck(False)
@cython.wraparound(False)
def levenshtein_one_vs_many(np.ndarray[np.int_t, ndim=1] a, np.ndarray[np.int_t, ndim=2] bs,
                            np.ndarray[np.float32_t, ndim=1] weights):
    cdef np.ndarray[np.int_t, ndim=2] sub_op = np.empty((4, 4), dtype=np.int)
    sub_op[0] = [-1, 4, 5, 6]
    sub_op[1] = [4, -1, 7, 8]
    sub_op[2] = [5, 7, -1, 9]
    sub_op[3] = [6, 8, 9, -1]

    cdef int l_a = a.shape[0]
    cdef int l_b = bs.shape[1]

    cdef np.ndarray[np.float32_t, ndim=1] prev_row = np.empty(l_b + 1, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] cur_row  = np.empty(l_b + 1, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] dists = np.empty(bs.shape[0], dtype=np.float32)

    cdef np.ndarray[np.float32_t, ndim=2] grad_prev_row = np.empty((l_b + 1, 10), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] grad_cur_row  = np.empty((l_b + 1, 10), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] grad_dists = np.empty((bs.shape[0], 10), dtype=np.float32)

    cdef int i, j, k
    # cdef np.ndarray[np.float32_t, ndim=1] values = np.empty(3, dtype=np.float32)
    cdef np.float32_t del_cost, ins_cost, sub_cost
    # cdef int best_op
    for k in range(bs.shape[0]):
        prev_row[0] = 0
        grad_prev_row[0] = 0
        for j in range(l_b):
            prev_row[j + 1] = prev_row[j] + weights[bs[k, j]]
            grad_prev_row[j + 1] = grad_prev_row[j]
            grad_prev_row[j + 1, bs[k, j]] += 1
        for i in range(l_a):
            cur_row[0] = prev_row[0] + weights[a[i]]
            grad_cur_row[0] = grad_prev_row[0]
            grad_cur_row[0, a[i]] += 1

            for j in range(l_b):
                del_cost = prev_row[j + 1] + weights[a[i]]
                ins_cost = cur_row[j] + weights[bs[k, j]]
                sub_cost = prev_row[j] if a[i] == bs[k, j] else prev_row[j] + weights[sub_op[a[i], bs[k, j]]]

                if del_cost < ins_cost and del_cost < sub_cost:
                    grad_cur_row[j + 1] = grad_prev_row[j + 1]
                    grad_cur_row[j + 1, a[i]] += 1
                    cur_row[j + 1] = del_cost
                elif ins_cost < sub_cost:
                    grad_cur_row[j + 1] = grad_cur_row[j]
                    grad_cur_row[j + 1, bs[k, j]] += 1
                    cur_row[j + 1] = ins_cost
                elif a[i] == bs[k, j]:
                    grad_cur_row[j + 1] = grad_prev_row[j]
                    cur_row[j + 1] = sub_cost
                else:
                    grad_cur_row[j + 1] = grad_prev_row[j]
                    grad_cur_row[j + 1, sub_op[a[i], bs[k, j]]] += 1
                    cur_row[j + 1] = sub_cost

            cur_row, prev_row = prev_row, cur_row
            grad_cur_row, grad_prev_row = grad_prev_row, grad_cur_row

        dists[k] = prev_row[l_b]
        grad_dists[k] = grad_prev_row[l_b]

    return dists, grad_dists.T


# noinspection PyUnresolvedReferences
@cython.boundscheck(False)
@cython.wraparound(False)
def levenshtein_one_vs_many_v2(np.ndarray[np.int_t, ndim=1] a, np.ndarray[np.int_t, ndim=2] bs,
                               np.ndarray[np.float32_t, ndim=1] weights):
    cdef np.ndarray[np.int_t, ndim=2] sub_op = np.empty((4, 4), dtype=np.int)
    sub_op[0] = [-1, 4, 5, 6]
    sub_op[1] = [4, -1, 7, 8]
    sub_op[2] = [5, 7, -1, 9]
    sub_op[3] = [6, 8, 9, -1]

    cdef int l_a = a.shape[0]
    cdef int l_b = bs.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2] partial_dists = np.empty((l_a + 1, l_b + 1), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] dists = np.empty(bs.shape[0], dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] grad_dists = np.zeros((bs.shape[0], 10), dtype=np.float32)

    cdef int i, j, k
    for k in range(bs.shape[0]):
        partial_dists[0] = 0
        for j in range(l_b):
            partial_dists[0, j + 1] = partial_dists[0, j] + weights[bs[k, j]]
        for i in range(l_a):
            partial_dists[i + 1, 0] = partial_dists[i, 0] + weights[a[i]]

            for j in range(l_b):
                partial_dists[i + 1, j + 1] = min(
                    partial_dists[i, j + 1] + weights[a[i]],
                    partial_dists[i + 1, j] + weights[bs[k, j]],
                    partial_dists[i, j] if a[i] == bs[k, j] else partial_dists[i, j] + weights[sub_op[a[i], bs[k, j]]]
                )

        dists[k] = partial_dists[l_a, l_b]

        i = l_a - 1
        j = l_b - 1
        while i >= 0 and j >= 0:
            del_cost = partial_dists[i, j + 1] + weights[a[i]]
            ins_cost = partial_dists[i + 1, j] + weights[bs[k, j]]
            sub_cost = partial_dists[i, j] if a[i] == bs[k, j] else partial_dists[i, j] + weights[sub_op[a[i], bs[k, j]]]

            if del_cost < ins_cost and del_cost < sub_cost:
                grad_dists[k, a[i]] += 1
                i -= 1
            elif ins_cost < sub_cost:
                grad_dists[k, bs[k, j]] += 1
                j -= 1
            elif a[i] == bs[k, j]:
                i -= 1
                j -= 1
            else:
                grad_dists[k, sub_op[a[i], bs[k, j]]] += 1
                i -= 1
                j -= 1
        while i >= 0:
            grad_dists[k, a[i]] += 1
            i -= 1
        while j >= 0:
            grad_dists[k, bs[k, j]] += 1
            j -= 1

    return dists, grad_dists.T

# noinspection PyUnresolvedReferences
@cython.boundscheck(False)
@cython.wraparound(False)
def local_alignment_one_vs_many(np.ndarray[np.int_t, ndim=1] a, np.ndarray[np.int_t, ndim=2] bs,
                                np.ndarray[np.float32_t, ndim=1] exp_bparams):
    cdef np.float32_t exp_bd = exp_bparams[0]
    cdef np.float32_t exp_be = exp_bparams[1]
    cdef np.ndarray[np.int_t, ndim=2] sub_op = np.empty((4, 4), dtype=np.int)
    sub_op[0] = [2,  3,  4,  5]
    sub_op[1] = [3,  6,  7,  8]
    sub_op[2] = [4,  7,  9, 10]
    sub_op[3] = [5,  8, 10, 11]

    cdef int l_a = a.shape[0]
    cdef int l_b = bs.shape[1]

    cdef np.ndarray[np.float32_t, ndim=1] M_prev_row = np.empty(l_b + 1, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] M_cur_row  = np.empty(l_b + 1, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] X_prev_row = np.empty(l_b + 1, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] X_cur_row  = np.empty(l_b + 1, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] Y_prev_row = np.empty(l_b + 1, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] Y_cur_row  = np.empty(l_b + 1, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] A_prev_row = np.empty(l_b + 1, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] A_cur_row  = np.empty(l_b + 1, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] B_prev_row = np.empty(l_b + 1, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] B_cur_row  = np.empty(l_b + 1, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] dists = np.empty(bs.shape[0], dtype=np.float32)

    cdef np.ndarray[np.float32_t, ndim=1] grad_d_M_prev_row = np.empty(l_b + 1, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] grad_d_M_cur_row  = np.empty(l_b + 1, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] grad_d_X_prev_row = np.empty(l_b + 1, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] grad_d_X_cur_row  = np.empty(l_b + 1, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] grad_d_Y_prev_row = np.empty(l_b + 1, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] grad_d_Y_cur_row  = np.empty(l_b + 1, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] grad_d_A_prev_row = np.empty(l_b + 1, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] grad_d_A_cur_row  = np.empty(l_b + 1, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] grad_d_B_prev_row = np.empty(l_b + 1, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] grad_d_B_cur_row  = np.empty(l_b + 1, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] grad_d_dists = np.empty(bs.shape[0], dtype=np.float32)

    cdef np.ndarray[np.float32_t, ndim=1] grad_e_M_prev_row = np.empty(l_b + 1, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] grad_e_M_cur_row  = np.empty(l_b + 1, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] grad_e_X_prev_row = np.empty(l_b + 1, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] grad_e_X_cur_row  = np.empty(l_b + 1, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] grad_e_Y_prev_row = np.empty(l_b + 1, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] grad_e_Y_cur_row  = np.empty(l_b + 1, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] grad_e_A_prev_row = np.empty(l_b + 1, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] grad_e_A_cur_row  = np.empty(l_b + 1, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] grad_e_B_prev_row = np.empty(l_b + 1, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] grad_e_B_cur_row  = np.empty(l_b + 1, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] grad_e_dists = np.empty(bs.shape[0], dtype=np.float32)

    cdef np.ndarray[np.float32_t, ndim=2] grad_s_M_prev_row = np.empty((l_b + 1, 10), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] grad_s_M_cur_row  = np.empty((l_b + 1, 10), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] grad_s_X_prev_row = np.empty((l_b + 1, 10), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] grad_s_X_cur_row  = np.empty((l_b + 1, 10), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] grad_s_Y_prev_row = np.empty((l_b + 1, 10), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] grad_s_Y_cur_row  = np.empty((l_b + 1, 10), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] grad_s_A_prev_row = np.empty((l_b + 1, 10), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] grad_s_A_cur_row  = np.empty((l_b + 1, 10), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] grad_s_B_prev_row = np.empty((l_b + 1, 10), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] grad_s_B_cur_row  = np.empty((l_b + 1, 10), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] grad_s_dists = np.empty((bs.shape[0], 10), dtype=np.float32)

    cdef int i, j, k
    for k in range(bs.shape[0]):
        M_prev_row.fill(0)
        X_prev_row.fill(0)
        Y_prev_row.fill(0)
        A_prev_row.fill(0)
        B_prev_row.fill(0)

        grad_d_M_prev_row.fill(0)
        grad_d_X_prev_row.fill(0)
        grad_d_Y_prev_row.fill(0)
        grad_d_A_prev_row.fill(0)
        grad_d_B_prev_row.fill(0)

        grad_e_M_prev_row.fill(0)
        grad_e_X_prev_row.fill(0)
        grad_e_Y_prev_row.fill(0)
        grad_e_A_prev_row.fill(0)
        grad_e_B_prev_row.fill(0)

        grad_s_M_prev_row.fill(0)
        grad_s_X_prev_row.fill(0)
        grad_s_Y_prev_row.fill(0)
        grad_s_A_prev_row.fill(0)
        grad_s_B_prev_row.fill(0)

        for i in range(l_a):
            M_cur_row[0] = 0
            X_cur_row[0] = 0
            Y_cur_row[0] = 0
            A_cur_row[0] = 0
            B_cur_row[0] = 0

            grad_d_M_cur_row[0] = 0
            grad_d_X_cur_row[0] = 0
            grad_d_Y_cur_row[0] = 0
            grad_d_A_cur_row[0] = 0
            grad_d_B_cur_row[0] = 0

            grad_e_M_cur_row[0] = 0
            grad_e_X_cur_row[0] = 0
            grad_e_Y_cur_row[0] = 0
            grad_e_A_cur_row[0] = 0
            grad_e_B_cur_row[0] = 0

            grad_s_M_cur_row[0] = 0
            grad_s_X_cur_row[0] = 0
            grad_s_Y_cur_row[0] = 0
            grad_s_A_cur_row[0] = 0
            grad_s_B_cur_row[0] = 0

            for j in range(l_b):
                M_cur_row[j + 1] = exp_bparams[sub_op[a[i], bs[k, j]]] * (1 + X_prev_row[j] + Y_prev_row[j] + M_prev_row[j])
                X_cur_row[j + 1] = exp_bd * M_prev_row[j + 1] + exp_be * X_prev_row[j + 1]
                Y_cur_row[j + 1] = exp_bd * (M_cur_row[j] + X_cur_row[j]) + exp_be * Y_cur_row[j]
                A_cur_row[j + 1] = M_prev_row[j + 1] + A_prev_row[j + 1]
                B_cur_row[j + 1] = M_cur_row[j] + A_cur_row[j] + B_cur_row[j]

                grad_d_M_cur_row[j + 1] = exp_bparams[sub_op[a[i], bs[k, j]]] * (grad_d_X_prev_row[j] + grad_d_Y_prev_row[j] + grad_d_M_prev_row[j])
                grad_d_X_cur_row[j + 1] = M_prev_row[j + 1] + exp_bd * grad_d_M_prev_row[j + 1] + exp_be * grad_d_X_prev_row[j + 1]
                grad_d_Y_cur_row[j + 1] = M_cur_row[j] + X_cur_row[j] + exp_bd * (grad_d_M_cur_row[j] + grad_d_X_cur_row[j]) + exp_be * grad_d_Y_cur_row[j]
                grad_d_A_cur_row[j + 1] = grad_d_M_prev_row[j + 1] + grad_d_A_prev_row[j + 1]
                grad_d_B_cur_row[j + 1] = grad_d_M_cur_row[j] + grad_d_A_cur_row[j] + grad_d_B_cur_row[j]

                grad_e_M_cur_row[j + 1] = exp_bparams[sub_op[a[i], bs[k, j]]] * (grad_e_X_prev_row[j] + grad_e_Y_prev_row[j] + grad_e_M_prev_row[j])
                grad_e_X_cur_row[j + 1] = exp_bd * grad_e_M_prev_row[j + 1] + X_prev_row[j + 1] + exp_be * grad_e_X_prev_row[j + 1]
                grad_e_Y_cur_row[j + 1] = exp_bd * (grad_e_M_cur_row[j] + grad_e_X_cur_row[j]) + Y_cur_row[j] + exp_be * grad_e_Y_cur_row[j]
                grad_e_A_cur_row[j + 1] = grad_e_M_prev_row[j + 1] + grad_e_A_prev_row[j + 1]
                grad_e_B_cur_row[j + 1] = grad_e_M_cur_row[j] + grad_e_A_cur_row[j] + grad_e_B_cur_row[j]

                grad_s_M_cur_row[j + 1] = exp_bparams[sub_op[a[i], bs[k, j]]] * (grad_s_X_prev_row[j] + grad_s_Y_prev_row[j] + grad_s_M_prev_row[j])
                grad_s_M_cur_row[j + 1, sub_op[a[i], bs[k, j]] - 2] += 1 + X_prev_row[j] + Y_prev_row[j] + M_prev_row[j]
                grad_s_X_cur_row[j + 1] = exp_bd * grad_s_M_prev_row[j + 1] + exp_be * grad_s_X_prev_row[j + 1]
                grad_s_Y_cur_row[j + 1] = exp_bd * (grad_s_M_cur_row[j] + grad_s_X_cur_row[j]) + exp_be * grad_s_Y_cur_row[j]
                grad_s_A_cur_row[j + 1] = grad_s_M_prev_row[j + 1] + grad_s_A_prev_row[j + 1]
                grad_s_B_cur_row[j + 1] = grad_s_M_cur_row[j] + grad_s_A_cur_row[j] + grad_s_B_cur_row[j]

            M_cur_row, M_prev_row = M_prev_row, M_cur_row
            X_cur_row, X_prev_row = X_prev_row, X_cur_row
            Y_cur_row, Y_prev_row = Y_prev_row, Y_cur_row
            A_cur_row, A_prev_row = A_prev_row, A_cur_row
            B_cur_row, B_prev_row = B_prev_row, B_cur_row

            grad_d_M_cur_row, grad_d_M_prev_row = grad_d_M_prev_row, grad_d_M_cur_row
            grad_d_X_cur_row, grad_d_X_prev_row = grad_d_X_prev_row, grad_d_X_cur_row
            grad_d_Y_cur_row, grad_d_Y_prev_row = grad_d_Y_prev_row, grad_d_Y_cur_row
            grad_d_A_cur_row, grad_d_A_prev_row = grad_d_A_prev_row, grad_d_A_cur_row
            grad_d_B_cur_row, grad_d_B_prev_row = grad_d_B_prev_row, grad_d_B_cur_row

            grad_e_M_cur_row, grad_e_M_prev_row = grad_e_M_prev_row, grad_e_M_cur_row
            grad_e_X_cur_row, grad_e_X_prev_row = grad_e_X_prev_row, grad_e_X_cur_row
            grad_e_Y_cur_row, grad_e_Y_prev_row = grad_e_Y_prev_row, grad_e_Y_cur_row
            grad_e_A_cur_row, grad_e_A_prev_row = grad_e_A_prev_row, grad_e_A_cur_row
            grad_e_B_cur_row, grad_e_B_prev_row = grad_e_B_prev_row, grad_e_B_cur_row

            grad_s_M_cur_row, grad_s_M_prev_row = grad_s_M_prev_row, grad_s_M_cur_row
            grad_s_X_cur_row, grad_s_X_prev_row = grad_s_X_prev_row, grad_s_X_cur_row
            grad_s_Y_cur_row, grad_s_Y_prev_row = grad_s_Y_prev_row, grad_s_Y_cur_row
            grad_s_A_cur_row, grad_s_A_prev_row = grad_s_A_prev_row, grad_s_A_cur_row
            grad_s_B_cur_row, grad_s_B_prev_row = grad_s_B_prev_row, grad_s_B_cur_row

        dists[k] = 1 + M_prev_row[l_b] + A_prev_row[l_b] + B_prev_row[l_b]
        grad_d_dists[k] = grad_d_M_prev_row[l_b] + grad_d_A_prev_row[l_b] + grad_d_B_prev_row[l_b]
        grad_e_dists[k] = grad_e_M_prev_row[l_b] + grad_e_A_prev_row[l_b] + grad_e_B_prev_row[l_b]
        grad_s_dists[k] = grad_s_M_prev_row[l_b] + grad_s_A_prev_row[l_b] + grad_s_B_prev_row[l_b]

    grad_dists = np.concatenate((grad_d_dists[None], grad_e_dists[None], grad_s_dists.T), axis=0)
    return dists, grad_dists
