import autograd as ag
import matplotlib.pyplot as plt
import numpy as np


def test_newton(plot=False):
    """ Checks that a single Newton iteration is sufficient to reach the
    solution of a quadratic problem. """
    u = np.random.random(3)
    v = np.random.random(3)
    w = np.random.random(3)
    A = ag.tensor(np.outer(u, u) + np.outer(v, v) + np.outer(w, w))
    b = ag.tensor(np.random.random(3))
    c = ag.tensor(np.random.random())

    def f(x):
        return A.dot(x).dot(x) / 2 + b.dot(x) + c

    x = ag.tensor(100 * np.ones(3), requires_grad=True)
    fs = []
    for _ in range(10):
        fx = f(x)
        fs.append(fx.data)

        g = fx.compute_grad(x.id)
        H = g.compute_grad(x.id)
        d = H.inv().dot(g)
        x = (x - d).detach()

    if plot:
        plt.plot(fs)
        plt.show()
    print('f*th = {}, f*nt = {}'.format(f(-A.inv().dot(b)), fs[-1]))


if __name__ == '__main__':
    test_newton()

    def f(a, b):
        x = a * b + a.exp()
        x = x @ x.T
        x = x.inv()
        x = (x ** 2).log() / b
        x = x.reshape(-1)[0:7:2].reshape((2, 2))
        x = x.tensordot(x, ([0, 1], [1, 0]))
        x = x[None]
        return x
    ag.test.check_gradients(f, ag.random((3, 3), requires_grad=True),
                               ag.random((3, 3), requires_grad=True))

    import os

    if not os.path.exists('data'):
        os.chdir('..')

    from utils import *
    from spectrum import *
    from newton import fit

    n = 500
    d = 10
    k = 2
    X = np.random.random((n, d)).astype(np.float32)
    Y = np.sin(X.sum(-1))
    folds = k_folds_indices(X.shape[0], k=k)

    def kernel(θ, I, J):
        K = ag.exp(-ag.tensor(np.sum((X[I, None] - X[None, J]) ** 2, axis=-1)) / θ[0].exp())
        return K

    def epoch(μ):
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
            Inv = ag.inv(K_train + λ * ag.eye(n))
            α = Inv.dot(Y_train)

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
        return err_valid


    λ = ag.zeros(1)
    θ = ag.zeros(1)
    μ = ag.concatenate((θ, λ)).detach(requires_grad=True)
    with ag.Config(debug=True):
        ag.test.check_gradients(epoch, μ)
