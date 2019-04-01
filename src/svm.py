import sys
import autograd as ag
import numpy as np
import cvxopt
from native_utils import coordinate_descent


class SVC:
    def __init__(self, kernel='precomputed', C=1, loss='hinge'):
        assert kernel == 'precomputed'
        self.C = ag.tensor(C)
        self.alpha = None
        self.loss_ = loss
        if loss not in ('hinge', 'squared_hinge'):
            raise ValueError("Unknown loss: {}.".format(loss))

    def fit(self, k, y):
        n = k.shape[0]
        y = y.astype(np.float32) * 2 - 1
        k, y = ag.tensors(k, y)

        if self.loss_ == 'hinge':
            self.alpha, _ = ag.qp(
                k,
                -y,
                ag.concatenate((ag.diagflat(y), -ag.diagflat(y))),
                ag.concatenate((self.C * ag.ones(n), ag.zeros(n))),
                options=dict(show_progress=False)
            )
        elif self.loss_ == 'squared_hinge':
            self.alpha, _ = ag.qp(
                k + ag.eye(n) / (2 * self.C),
                -y,
                -ag.diagflat(y),
                ag.zeros(n),
                options=dict(show_progress=False)
            )

        return self

    def predict(self, k):
        k = ag.tensor(k)
        return k.dot(self.alpha).data >= 0.


class SVCIntercept:
    def __init__(self, kernel='precomputed', C=1, loss='hinge'):
        assert kernel == 'precomputed'
        self.C = C
        self.alpha = None
        self.intercept = None
        self.loss = loss
        if loss not in ('hinge', 'squared_hinge'):
            raise ValueError("Unknown loss: {}.".format(loss))

        self.eps = 1e-3

    def fit(self, k, y):
        n = k.shape[0]
        y = y.astype(np.float) * 2 - 1

        if self.loss == 'hinge':
            P = k
            q = -y
            G = np.concatenate((np.diag(y), -np.diag(y)))
            h = np.concatenate((self.C * np.ones(n), np.zeros(n)))
            A = np.ones((1, n))
            b = np.zeros(1)
            self.alpha = np.array(cvxopt.solvers.qp(*map(lambda c: cvxopt.matrix(c.astype(np.float64)),
                                                         (P, q, G, h, A, b)),
                                                    options=dict(show_progress=False))['x']).reshape(-1)
            low = min(self.C * self.eps, 1e-2)
            high = self.C * (1 - self.eps)
            alpha = self.alpha * y
            support_vectors = np.logical_and(alpha > low, alpha < high)

            if np.sum(support_vectors) == 0:
                # Degenerate case.
                alpha[alpha < low] = np.inf
                support_vectors = np.argmin(alpha)
                print("Degenerate intercept.", file=sys.stderr)
            self.intercept = np.mean(y[support_vectors] - k[support_vectors, :].dot(self.alpha))

        elif self.loss == 'squared_hinge':
            P = k + np.eye(n) / (2 * self.C)
            q = -y
            G = -np.diag(y)
            h = np.zeros(n)
            A = np.ones((1, n))
            b = np.zeros(1)
            self.alpha = np.array(cvxopt.solvers.qp(*map(lambda c: cvxopt.matrix(c.astype(np.float64)),
                                                         (P, q, G, h, A, b)),
                                                    options=dict(show_progress=False))['x']).reshape(-1)

            low = min(self.C * self.eps, 1e-2)
            support_vectors = self.alpha * y > low
            self.intercept = np.mean(y[support_vectors] - self.alpha[support_vectors]/2/self.C - k[support_vectors, :].dot(self.alpha))

        return self

    def predict(self, k):
        return k.dot(self.alpha) + self.intercept >= 0.


class SVCCoordinate:
    def __init__(self, kernel='precomputed', C=1, loss='hinge', intercept=0, **params):
        assert kernel == 'precomputed'
        self.C = C
        self.alpha = None
        self.loss = loss
        self.params = params
        self.intercept = intercept

        if loss not in ('hinge', 'squared_hinge'):
            raise ValueError("Unknown loss: {}.".format(loss))

    def fit(self, k, y):
        self.alpha = coordinate_descent(k=(k+self.intercept).astype(np.float32), y=y.astype(np.int),
                                        C=float(self.C), loss=self.loss, **self.params)
        return self

    def predict(self, k):
        k = ag.tensor(k+self.intercept)
        return k.dot(self.alpha).data >= 0.
