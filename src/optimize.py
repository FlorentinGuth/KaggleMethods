import autograd as ag
import tqdm
import svm
import numpy as np


class KernelRidge:
    def __init__(self, params):
        self.λ = ag.exp(params[0])
        self.α = None

    def fit(self, K, y):
        self.α = ag.solve(K + self.λ * ag.eye(len(y)), y, hermitian=True)

    def predict(self, K):
        return K.dot(self.α) > .5

    def loss(self, K, y):
        return ag.mean((K.dot(self.α) - y) ** 2)


class SVM(svm.SVC):
    def __init__(self, params):
        super().__init__(C=ag.exp(params[0]))

    def predict(self, K):
        return ag.tensor(super().predict(K))

    def loss(self, K, y):
        def hinge(z):
            return ag.maximum(1 - z, 0)
        return hinge(ag.tensor(y) * K.dot(self.alpha)).mean()


def optimize(kernel, clf, Y, indices, folds, θ, λ, β=1., iters=100, verbose=True):
    """ Gradient descent on the kernel parameters and classifier hyper-parameters to optimize the validation loss.
    :param kernel: function from θ and indices to kernel matrix (shape NxN)
    :param clf: classifier class to fit
    :param Y: the labels to predict (shape N)
    :param indices: generates a list of indices to use for the current step
    :param folds: generates a list of random folds (list of (train_idx, valid_idx) pairs)
    :param θ: the kernel hyper-parameters (1D array)
    :param λ: the classifier hyper-parameters (1D array)
    :param β: learning rate
    :param iters: number of iterations to do
    :param verbose: print steps
    :return: θ, λ, stats (list of (err_train, err_valid, acc_train, acc_valid))
    """
    p = len(θ)
    μ = ag.concatenate((θ, λ)).astype(np.float32).detach(requires_grad=True)
    Y = ag.tensor(Y).astype(np.float32)

    progress = tqdm.tqdm(range(iters))
    stats = []

    def epoch():
        θ, λ = μ[:p], μ[p:]
        C = clf(λ)

        err_trains = []
        err_valids = []
        acc_trains = []
        acc_valids = []

        def err_acc(K, Y):
            err = C.loss(K, Y)
            assert err.dtype == np.float32, err.dtype
            acc = ag.mean((C.predict(K) == Y))
            return err, acc

        idx = indices()
        K_idx = kernel(θ, idx)
        assert K_idx.dtype == np.float32, K_idx.dtype
        Y_idx = Y[idx]
        for I_train, I_valid in folds(len(idx)):
            Y_train = Y_idx[I_train]
            K_train = K_idx[np.ix_(I_train, I_train)]
            C.fit(K_train, Y_train)
            err_train, acc_train = err_acc(K_train, Y_train)
            err_trains.append(err_train)
            acc_trains.append(acc_train)

            Y_valid = Y_idx[I_valid]
            K_valid = K_idx[np.ix_(I_valid, I_train)]
            err_valid, acc_valid = err_acc(K_valid, Y_valid)
            err_valids.append(err_valid)
            acc_valids.append(acc_valid)

        err_train = ag.mean(err_trains)
        err_valid = ag.mean(err_valids)
        acc_train = ag.mean(acc_trains)
        acc_valid = ag.mean(acc_valids)
        stats.append((err_train.detach(), err_valid.detach(), acc_train.detach(), acc_valid.detach()))
        progress.desc = 'acc={:.2f}'.format(acc_valid)

        with ag.Config(grad=False):
            g = err_valid.compute_grad(μ.id)
            Δ = -β * g
            if verbose:
                print('Θ norm {:.1e}, err_train {:.1e}, err_valid {:.1e}, acc_train {:.0f}%, acc_valid {:.0f}%, g norm {:.2e}, Δ norm {:.2e}'
                  .format(ag.test.norm(θ), err_train, err_valid, 100*acc_train, 100*acc_valid, ag.test.norm(g), ag.test.norm(Δ)))
        return (μ + Δ).detach()

    for _ in progress:
        μ = epoch()

    return μ[:p], μ[p:], stats


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
        spectrum_K = np.sum([kernels[0][0] for kernels in spectrum_kernels], axis=0).astype(float)
        del spectrum_kernels

        # print(ag.test.summary(spectrum_K))
        fake_K = ag.ones((len(spectrum_K), len(spectrum_K)))

        K = ag.stack((spectrum_K, fake_K)).astype(np.float32)

        def spectrum_sum(θ, I):
            return ag.tensordot(K[np.index_exp[:] + np.ix_(I, I)], ag.exp(θ), axes=([0], [0]))

        n = 200
        num_folds = 2
        θ = ag.tensor([-10, 0])
        λ = ag.zeros(1)

        θ, λ, stats = optimize(
            kernel=spectrum_sum,
            clf=SVM,
            Y=train_Ys[0],
            indices=lambda: np.arange(n),
            folds= lambda n: k_folds_indices(n, num_folds),
            θ=θ,
            λ=λ,
            β=1e2,
            iters=100,
        )
        print(θ, λ)

    run()
