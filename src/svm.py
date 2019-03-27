import autograd as ag
import numpy as np


class SVC:
    def __init__(self, kernel='precomputed', C=1):
        assert kernel == 'precomputed'
        self.C = C
        self.alpha = None

    def fit(self, k, y):
        n = k.shape[0]
        y = y.astype(np.float) * 2 - 1
        k, y = ag.tensors(k, y)
        self.alpha, _ = ag.qp(
            k,
            -y,
            ag.tensor(np.concatenate((np.diag(y.data), -np.diag(y.data)))),
            ag.tensor(np.concatenate((np.full(n, self.C), np.zeros(n)))),
            options=dict(show_progress=False)
        )

    def predict(self, k):
        k = ag.tensor(k)
        return (k * self.alpha[None, :, 0]).sum(axis=1).data >= 0.
