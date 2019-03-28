import autograd as ag
import numpy as np


class SVC:
    def __init__(self, kernel='precomputed', C=1):
        assert kernel == 'precomputed'
        self.C = ag.tensor(C)
        self.alpha = None

    def fit(self, k, y):
        n = k.shape[0]
        y = y.astype(np.float) * 2 - 1
        k, y = ag.tensors(k, y)
        self.alpha, _ = ag.qp(
            k,
            -y,
            ag.concatenate((np.diag(y.data), -np.diag(y.data))),
            ag.concatenate((self.C * np.ones(n), np.zeros(n))),
            options=dict(show_progress=False)
        )

    def predict(self, k):
        k = ag.tensor(k)
        return k.dot(self.alpha).data >= 0.
