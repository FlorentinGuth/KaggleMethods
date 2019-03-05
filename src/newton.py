import autograd as ag
import numpy as np
import tqdm


def fit(kernel, Y, folds, θ, iters=100):
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

        for fold in folds:
            I_train, I_valid = fold
            Y_train = Y[I_train]
            Y_valid = Y[I_valid]
            n = len(I_train)

            K_train = kernel(θ, I_train, I_train)
            α = ag.solve(K_train + λ * ag.eye(n), Y_train, hermitian=True)

            err_train, acc_train = err_acc(K_train, α, Y_train)
            err_trains.append(err_train)
            acc_trains.append(acc_train)

            K_valid = kernel(θ, I_valid, I_train)
            err_valid, acc_valid = err_acc(K_valid, α, Y_valid)
            err_valids.append(err_valid)
            acc_valids.append(acc_valid)

        err_train = ag.mean(err_trains)
        err_valid = ag.mean(err_valids)
        acc_train = np.mean(acc_trains)
        acc_valid = np.mean(acc_valids)
        stats.append((err_train.detach(), err_valid.detach(), acc_train, acc_valid))
        progress.desc = 'acc={:.2f}'.format(acc_valid)

        g = err_valid.compute_grad(μ.id)
        H = g.compute_grad(μ.id)
        Δ = -ag.solve(H, g)
        return (μ + Δ).detach()

    for _ in progress:
        μ = epoch()

    return θ, λ, stats
