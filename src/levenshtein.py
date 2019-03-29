import native_utils
import autograd as ag
from tqdm import tqdm as tqdm_notebook
import numpy as np
from concurrent.futures import ProcessPoolExecutor as Executor
import multiprocessing


def parallel_dists(dist_fn, weights, X, Y=None, tqdm=False):
    X = ag.tensor(X.astype(np.int))
    if Y is None:
        Y = X
    else:
        Y = ag.tensor(Y.astype(np.int))
    weights = ag.tensor(weights)

    Ys = [Y[i:].data for i in range(len(X))]
    d = np.zeros((len(X), len(Y)), dtype=np.float32)
    g = np.empty((len(weights), len(X), len(Y)), dtype=np.float32)
    with Executor(max_workers=multiprocessing.cpu_count()) as executor:
        iterator = enumerate(executor.map(dist_fn, X.data, Ys, [weights.data] * len(X)))
        if tqdm:
            iterator = tqdm_notebook(iterator, total=len(X))
        for i, (row, grad_row) in iterator:
            d[i, i:] = row
            d[i, :i] = d[:i, i]
            g[:, i, i:] = grad_row
            g[:, i, :i] = g[:, :i, i]

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


def main():
    import os
    if not os.path.exists('data'):
        os.chdir('..')
    import utils
    import optimize
    import svm

    for dataset in [0, 1, 2]:
        print('DATASET={}'.format(dataset))
        X = utils.load(k=dataset)
        spec_k = utils.precomputed_kernels(None, 'cum_spectrum_31')[0][dataset]

        def levenshtein_kernel_diff(params, I):
            factors = ag.exp(params)
            dists = levenshtein_distance_v2(X[I], X[I], weights=factors[:10], tqdm=False)
            scale = factors[10]
            return ag.exp(- dists / (dists.mean() + 1e-3) * scale) + factors[11] * spec_k[I][:, J]

        n = 64
        num_folds = 2
        θ = ag.zeros(12)
        λ = ag.zeros(1)

        θ, λ, stats = optimize.optimize(
            kernel=levenshtein_kernel_diff,
            clf=optimize.KernelRidge,
            Y=utils.train_Ys[dataset],
            indices=lambda: np.random.permutation(len(X))[:n],
            fold_generator=lambda p: utils.k_folds_indices(p, num_folds)[:1],
            θ=θ,
            λ=λ,
            β=1e0,
            iters=500,
            verbose=False,
        )
        print(θ, λ)

        K = levenshtein_kernel_diff(θ, np.arange(len(X)), np.arange(len(X))).data
        print(utils.evaluate(
            svm.SVC(C=np.exp(λ.data[0])),
            K,
            utils.train_Ys[dataset],
            folds=20
        ))
        print(utils.evaluate(
            svm.SVC(C=np.exp(λ.data[0])),
            K,
            utils.train_Ys[dataset],
            folds=20
        ))
        print(utils.evaluate(
            svm.SVC(C=np.exp(λ.data[0])),
            K,
            utils.train_Ys[dataset],
            folds=20
        ))


if __name__ == '__main__':
    main()
