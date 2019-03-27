from .tensor import Tensor
import numpy as np
from copy import deepcopy


def norm(x):
    if isinstance(x, Tensor):
        x = x.data
    return np.linalg.norm(x.reshape(-1))


def err_rel(x, y, ε=1e-10):
    """ Relative error of y with respect to x. """
    return norm(x - y) / (norm(x) + ε)


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
    return grad


def check_gradients(f, *args, order=2, h=1e-4):
    """ Prints the relative error between the empirical and analytical gradients of the result of f(*args).
    order controls the number of times the result should be differentiated (1 for gradient, 2 for hessian...).
    """
    n = len(args)
    functions = [("Order", f)]
    for k in range(1, order + 1):
        grads = []
        for name, func in functions:
            for i in range(n):
                name_i = "{} \\partial_{}".format(name, i)
                grad = func(*args).compute_grad(args[i].id)
                grad2 = gradh(func, args, i, h)
                print("{}:\t{:.2e}".format(name_i, err_rel(grad2, grad.data)))
                # Use i=i forces python to capture by value when building the closure.
                grads.append((name_i, lambda *args, i=i, func=func: func(*args).compute_grad(args[i].id)))
        functions = grads


def build_graph(tensor, name):
    import graphviz

    dot = graphviz.Digraph()
    builded = {}
    size = 0

    def build_graph_rec(dot, node):
        nonlocal size
        if id(node) not in builded:
            builded[id(node)] = True
            name = node.grad_fn.__name__[len('grad_'):] if node.grad_fn is not None else 'None'
            dot.node(str(id(node)), '{} {}'.format(name, node.shape))
            sz = 4
            for s in node.shape:
                sz *= s
            size += sz
            for child in node.children:
                build_graph_rec(dot, child)
                dot.edge(str(id(node)), str(id(child)))

    build_graph_rec(dot, tensor)
    dot.save('{}.gv'.format(name))
    print('Size of {} is {:.0f} Mo'.format(name, size / 1024 / 1024))
