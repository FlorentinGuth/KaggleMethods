import autograd as ag


def fit(kernel, X, Y, folds, θ):
    λ =ag.tensor(1)
    μ = θ# μ = ag.concatenate((θ, ag.tensor(λ)[None])).detach()
    while True:
        θ = μ# θ, λ = μ[:-1], μ[-1]
        print(θ, λ)
        e_trains = []
        e_valids = []

        for fold in folds:
            I_train, I_valid = fold
            X_train, Y_train = X[I_train], Y[I_train]
            X_valid, Y_valid = X[I_valid], Y[I_valid]
            n = len(I_train)

            K_train = kernel(θ, I_train, I_train)
            Inv = ag.inv(K_train + λ * ag.eye(n))
            α = Inv.dot(Y_train)
            e_train = λ * α.dot(Y_train)
            e_trains.append(e_train)

            K_valid = kernel(θ, I_valid, I_train)
            Z_valid = K_valid.dot(α)
            e_valid = ag.sum((Z_valid - Y_valid) ** 2)
            e_valids.append(e_valid)

        e_train = ag.mean(e_trains)
        e_valid = ag.mean(e_valids)
        print(e_train, e_valid)

        g = e_valid.compute_grad(μ.id)
        print('g', np.linalg.norm(g.data))
        H = g.compute_grad(μ.id)
        print('H', np.linalg.norm(H.data))
        Δ = -ag.inv(H).dot(g)
        print('Δ', np.linalg.norm(Δ.data))
        μ = (μ + Δ).detach()


import os
if not os.path.exists('data'):
    os.chdir('..')
from utils import *
from spectrum import *
import numpy as np

n = 1000
d = 10
# X = load()[:n]
# Y = 2*load(X=False)[:n].astype(float)-1
X = np.random.random((n, d))
Y = np.sin(X.sum(-1))
folds = k_folds_indices(X.shape[0], k=5)
# K =  ag.tensor([k_spectrum(X, k=k) for k in range(1, 8)])
def kernel(θ, I, J):
    # k = K[:, I][:, :, J].detach()
    # return ag.tensordot(k, θ, ([0], [0]))
    K = ag.exp(-ag.tensor(np.sum((X[I,None] - X[None,J])**2, axis=-1))/θ[0])
    return K
θ = ag.ones(1)
fit(kernel, X, Y, folds, θ)
