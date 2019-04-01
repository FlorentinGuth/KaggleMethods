import data
import native_utils
import autograd as ag
from tqdm import tqdm as tqdm_notebook
import numpy as np
from concurrent.futures import ProcessPoolExecutor as Executor
import multiprocessing

from data import precomputed_kernels, transform_kernels


def parallel_dists(dist_fn, weights, X, Y=None, tqdm=False):
    X = ag.tensor(X.astype(np.int))
    if Y is None:
        Y = X
        Ys = [Y[i:].data for i in range(len(X))]
    else:
        Y = ag.tensor(Y.astype(np.int))
        Ys = [Y.data for i in range(len(X))]
    weights = ag.tensor(weights)

    d = np.zeros((len(X), len(Y)), dtype=np.float32)
    g = np.empty((len(weights), len(X), len(Y)), dtype=np.float32)
    with Executor(max_workers=multiprocessing.cpu_count()) as executor:
        iterator = enumerate(executor.map(dist_fn, X.data, Ys, [weights.data] * len(X)))
        if tqdm:
            iterator = tqdm_notebook(iterator, total=len(X))
        for i, (row, grad_row) in iterator:
            if len(row) < len(X):
                d[i, i:] = row
                d[i, :i] = d[:i, i]
                g[:, i, i:] = grad_row
                g[:, i, :i] = g[:, :i, i]
            else:
                d[i] = row
                g[:, i] = grad_row

    g_tensor = ag.Tensor(g, None, children=[])

    def grad_d(leaf_id):
        return ag.tensordot(weights.compute_grad(leaf_id), g_tensor, axes=([-1], [0]))

    return ag.Tensor(d, grad_d, children=[weights])


def levenshtein_distance(X, Y=None, weights=None, tqdm=False):
    return parallel_dists(native_utils.levenshtein_one_vs_many, weights, X, Y, tqdm)


def levenshtein_distance_v2(X, Y=None, weights=None, tqdm=False):
    return parallel_dists(native_utils.levenshtein_one_vs_many_v2, weights, X, Y, tqdm)


def local_alignment_kernel(X, Y=None, weights=None, tqdm=False, beta=0.5):
    return parallel_dists(native_utils.local_alignment_one_vs_many, np.exp(beta * weights), X, Y, tqdm)


def edit_kernel(kernel='gaussian', scale=1, d=1):
    # TODO: Is there a way not to use tensors?
    ins_del = .35
    sub_easy = .0626
    sub_hard = .3009
    edit_distances = precomputed_kernels(levenshtein_distance_v2, 'levenshtein_distance', max_workers=1,
                                         weights=np.array([ins_del, ins_del, ins_del, ins_del,
                                                           sub_hard, sub_easy, sub_hard, sub_hard, sub_easy, sub_hard],
                                                          np.float32))
    edit_distances = transform_kernels([edit_distances], lambda _, D: D.data)

    if kernel == 'gaussian':
        return transform_kernels([edit_distances], lambda i, K: np.exp(-(K/scale) ** 2))
    elif kernel == 'exp':
        return transform_kernels([edit_distances], lambda i, K: np.exp(-K/scale))
    elif kernel == 'polynomial':
        return transform_kernels([edit_distances], lambda i, K: 1 / (1 + (K/scale) ** d))
    else:
        raise ValueError("Unknown kernel.")


def main():
    import os
    if not os.path.exists('data'):
        os.chdir('..')
    import evaluation
    import optimize
    import svm

    for dataset in [0, 1, 2]:
        print('DATASET={}'.format(dataset))
        X = data.load(k=dataset)
        spec_k = data.precomputed_kernels(None, 'cum_spectrum_31')[0][dataset]

        def levenshtein_kernel_diff(params, I):
            factors = ag.exp(params)
            dists = levenshtein_distance_v2(X[I], X[I], weights=factors[:10], tqdm=False)
            scale = factors[10]
            return ag.exp(- dists / (dists.mean() + 1e-3) * scale) + factors[11] * spec_k[I][:, I].astype(np.float32)

        n = 512
        num_folds = 2
        θ = ag.zeros(12)
        λ = ag.zeros(1)

        θ, λ, stats = optimize.optimize(
            kernel=levenshtein_kernel_diff,
            clf=optimize.KernelRidge,
            Y=data.train_Ys[dataset],
            indices=lambda: np.random.permutation(len(X))[:n],
            folds=lambda p: data.k_folds_indices(p, num_folds),
            θ=θ,
            λ=λ,
            β=1e2,
            iters=50,
            verbose=False,
        )
        print(θ, λ)

        K = levenshtein_kernel_diff(θ, np.arange(len(X))).data
        for _ in range(3):
            print(evaluation.evaluate(
                svm.SVC(C=10),
                K,
                data.train_Ys[dataset],
                folds=20
            ))


if __name__ == '__main__':
    main()
