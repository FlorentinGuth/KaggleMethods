import autograd as ag
import tqdm
import svm


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


def optimize(kernel, clf, Y, num_folds, θ, λ, β=1, iters=100):
    """ Gradient descent on the kernel parameters and classifier hyper-parameters to optimize the validation loss.
    :param kernel: function from θ to kernel matrix (shape NxN)
    :param clf: classifier class to fit
    :param Y: the labels to predict (shape N)
    :param num_folds: number of folds in the cross-validation.
    :param θ: the kernel hyper-parameters (1D array)
    :param λ: the classifier hyper-parameters (1D array)
    :param β: learning rate
    :param iters: number of iterations to do
    :return: θ, λ, stats (list of (err_train, err_valid, acc_train, acc_valid))
    """
    p = len(θ)
    μ = ag.concatenate((θ, λ)).detach(requires_grad=True)

    progress = tqdm.tqdm_notebook(range(iters))
    stats = []

    def epoch():
        θ, λ = μ[:p], μ[p:]
        K = kernel(θ)
        C = clf(λ)

        err_trains = []
        err_valids = []
        acc_trains = []
        acc_valids = []

        def err_acc(K, Y):
            return C.loss(K, Y), ag.mean((C.predict(K) == Y))

        for fold in k_folds_indices(Y.shape[0], num_folds):
            I_train, I_valid = fold
            Y_train = Y[I_train]
            Y_valid = Y[I_valid]

            K_train = K[np.ix_(I_train, I_train)]
            C.fit(K_train, Y_train)

            with ag.Config(grad=False):
                err_train, acc_train = err_acc(K_train, Y_train)
                err_trains.append(err_train)
                acc_trains.append(acc_train)

            K_valid = K[np.ix_(I_valid, I_train)]
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
        spectrum_K = np.stack([kernels[0][0] for kernels in spectrum_kernels]).astype(float)
        del spectrum_kernels

        K = ag.tensor(spectrum_K)
        def spectrum_sum(θ):
            return ag.tensordot(K, ag.exp(θ), axes=([0], [0]))

        n = 200
        θ = ag.zeros(spectrum_K.shape[0])
        λ = ag.zeros(1)

        θ, λ, stats = optimize(
            kernel=spectrum_sum,
            clf=KernelRidge,
            Y=train_Ys[0].astype(float)[:n],
            num_folds=2,
            θ=θ,
            λ=λ,
            β=10,
            iters=100,
        )
        print(θ, λ)

    run()
