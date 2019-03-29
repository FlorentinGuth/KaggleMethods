import native_utils
import autograd as ag
from tqdm import tqdm as tqdm_notebook
import numpy as np
from concurrent.futures import ProcessPoolExecutor as Executor
import multiprocessing


def parallel_dists(dist_fn, weights, X, Y=None, tqdm=False):
    X = ag.tensor(X).astype(np.int)
    if Y is None:
        Y = X
    else:
        Y = ag.tensor(Y).astype(np.int)
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


def levenshtein_distance(weights, X, Y=None, tqdm=False):
    return parallel_dists(native_utils.levenshtein_one_vs_many, weights, X, Y, tqdm)


def levenshtein_distance_v2(weights, X, Y=None, tqdm=False):
    return parallel_dists(native_utils.levenshtein_one_vs_many_v2, weights, X, Y, tqdm)


def local_alignment_kernel(weights, X, Y=None, tqdm=False, beta=0.5):
    return parallel_dists(native_utils.local_alignment_one_vs_many, np.exp(beta * weights), X, Y, tqdm)


def main():
    import utils
    import optimize
    import svm

    dataset = 0

    X = utils.load(k=dataset)

    def levenshtein_kernel(params, I, J):
        factors = ag.exp(params)
        dists = levenshtein_distance_v2(factors[:10], X[I], X[J], tqdm=False)
        scale = factors[10]
        return ag.exp(- dists / (dists.mean() + 1e-3) * scale)

    n = 100
    num_folds = 2
    θ = ag.zeros(11)
    λ = ag.zeros(1)

    θ, λ, stats = optimize.optimize(
        kernel=levenshtein_kernel,
        clf=optimize.SVM,
        Y=utils.train_Ys[dataset].astype(float),
        fold_generator=lambda: utils.k_folds_indices(len(X), num_folds, n)[:1],
        θ=θ,
        λ=λ,
        β=2e1,
        iters=50,
    )
    print(θ, λ)

    print(utils.evaluate(
        svm.SVC(C=np.exp(λ.data[0])),
        levenshtein_kernel(θ, np.arange(len(X)), np.arange(len(X))).data,
        utils.train_Ys[dataset],
        folds=10
    ))


if __name__ == '__main__':
    main()
