import autograd as ag
import numpy as np
import tqdm


def fit(kernel, Y, num_folds, θ, β=1, iters=100):
    """ Parametric kernel ridge regression. Does a gradient descent on the kernel parameters (and the regularization)
    to optimize the validation error ||Y_pred - Y_true||².
    :param kernel: function from θ to kernel matrix (shape NxN)
    :param Y: the labels to predict (shape N)
    :param num_folds: number of folds in the cross-validation.
    :param θ: the parameters
    :param β: learning rate
    :param iters: number of iterations to do
    :return: θ, λ (regularization), stats (list of (err_train, err_valid, acc_train, acc_valid))
    """
    λ = ag.zeros(()) # float32 scalar 0
    μ = ag.concatenate((θ, λ[None])).detach(requires_grad=True)

    progress = tqdm.tqdm_notebook(range(iters))
    stats = []

    def epoch():
        θ, λ = μ[:-1], ag.exp(μ[-1])
        err_trains = []
        err_valids = []
        acc_trains = []
        acc_valids = []

        def err_acc(K, α, Y):
            Z = K.dot(α)
            P = Z.data > .5
            return ag.mean((Z - Y) ** 2), np.mean((P == Y.data))

        K = kernel(θ)
        for fold in k_folds_indices(Y.shape[0], num_folds):
            I_train, I_valid = fold
            Y_train = Y[I_train]
            Y_valid = Y[I_valid]
            n = len(I_train)

            K_train = K[np.ix_(I_train, I_train)]
            α = ag.solve(K_train + λ * ag.eye(n), Y_train, hermitian=True)

            with ag.Config(grad=False):
                err_train, acc_train = err_acc(K_train, α, Y_train)
                err_trains.append(err_train)
                acc_trains.append(acc_train)

            K_valid = K[np.ix_(I_valid, I_train)]
            err_valid, acc_valid = err_acc(K_valid, α, Y_valid)
            err_valids.append(err_valid)
            acc_valids.append(acc_valid)

        err_train = ag.mean(err_trains)
        err_valid = ag.mean(err_valids)
        acc_train = np.mean(acc_trains)
        acc_valid = np.mean(acc_valids)
        stats.append((err_train.detach(), err_valid.detach(), acc_train, acc_valid))
        progress.desc = 'acc={:.2f}'.format(acc_valid)

        with ag.Config(grad=False):
            g = err_valid.compute_grad(μ.id)
            Δ = -β * g
            print('Θ norm {:.1e}, err_train {:.1e}, err_valid {:.1e}, acc_train {:.0f}%, acc_valid {:.0f}%, g norm {:.2e}, Δ norm {:.2e}'
                  .format(ag.test.norm(θ.data), err_train.data, err_valid.data, 100*acc_train, 100*acc_valid, ag.test.norm(g.data), ag.test.norm(Δ.data)))
        return (μ + Δ).detach()

    for _ in progress:
        μ = epoch()

    return μ[:-1], ag.exp(μ[-1]), stats


if __name__ == '__main__':
    import os

    if not os.path.exists('data'):
        os.chdir('..')
    from spectrum import *
    from utils import *

    def run():
        spectrum_kernels = []
        for k in range(1, 14):
            spectrum_kernels.append(precomputed_kernels(k_spectrum, 'spectrum_{}'.format(k), k=k))
        spectrum_K = np.stack([kernels[0][0] for kernels in spectrum_kernels]).astype(float)
        del spectrum_kernels

        K = ag.tensor(spectrum_K)
        def spectrum_sum(θ):
            return ag.tensordot(K, ag.exp(θ), axes=([0], [0]))

        n = 2000
        θ = ag.zeros(spectrum_K.shape[0])
        print(θ)

        import tracemalloc
        tracemalloc.start()
        θ, λ, stats = fit(
            kernel=spectrum_sum,
            Y=train_Ys[0].astype(float)[:n],
            num_folds=2,
            θ=θ,
            β=10,
            iters=3,
        )
        print(θ, λ)
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        print("[ Top 10 ]")
        for stat in top_stats[:10]:
            print(stat)

    run()
