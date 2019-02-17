import autograd as ag
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy


def test_newton():
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

    x = ag.tensor(100 * np.ones(3))
    fs = []
    for _ in range(10):
        fx = f(x)
        fs.append(fx.data)

        g = fx.compute_grad(x.id)
        H = g.compute_grad(x.id)
        d = H.inv().dot(g)
        x = (x - d).detach()

    plt.plot(fs)
    plt.show()
    print('f*th = {}, f*nt = {}'.format(f(-A.inv().dot(b)), fs[-1]))


def test_grad():
    """ Compares empirical gradients and hessians to the automatically computed versions. """

    def f(a, b):
        x = a * b + a.exp()
        x = x @ x.T
        x = x.inv()
        x = (x**2).log() / b
        x = x.reshape(-1)[0:7:2].reshape((2, 2))
        x = x.tensordot(x, ([0, 1], [1, 0]))
        x = x[None]
        return x

    def gradh(f, args, i, h=1e-4):
        """ Empirical gradient of f(*args) with respect to args[i]. """
        res = f(*args)
        grad = np.empty(args[i].shape + res.shape)
        for index in np.ndindex(args[i].shape):
            argsh = list(deepcopy(args))
            argsh[i].data[index] = args[i].data[index] + h
            argsh2 = list(deepcopy(args))
            argsh2[i].data[index] = args[i].data[index] - h
            grad[index] = (f(*argsh).data - f(*argsh2).data) / (2 * h)
        return ag.tensor(grad)

    def norm(x):
        return np.linalg.norm(x.reshape(-1))

    args = (ag.tensor(np.random.random((3, 3))),
            ag.tensor(np.random.random((3, 3))))
    res = f(*args)
    n = len(args)
    for i in range(n):
        grad = res.compute_grad(args[i].id)
        grad2 = gradh(f, args, i)
        print(norm(grad.data - grad2.data) / norm(grad2.data))

        for j in range(n):
            hess = grad.compute_grad(args[j].id)
            hess2 = gradh(lambda *args: gradh(f, args, i), args, j)
            print(norm(hess.data - hess2.data) / norm(hess2.data))


if __name__ == '__main__':
    test_grad()

