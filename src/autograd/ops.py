from . import tensor as t
import numpy as np

leaves = []  # Array of shapes of leaves


def leaf(x):
    ''' A leaf is a tensor whose gradient to itself is the constant I.
    As a user, you should not use this function (use tensor() instead).
    '''
    x = np.array(x)
    x_id = len(leaves)
    leaves.append(x.shape)

    def grad_leaf(leaf_id):
        if leaf_id == x_id:
            return leaf(np.eye(x.size).reshape(x.shape + x.shape))
        else:
            return leaf(np.zeros(leaves[leaf_id] + x.shape))

    return t.Tensor(x, grad_leaf, x_id)


def tensor(x):
    ''' Makes sure x is a t. '''
    if isinstance(x, t.Tensor):
        return x
    else:
        return leaf(x)


def tensors(*args):
    return (tensor(a) for a in args)


def samedims(*args):
    # Prepend 1s to the shape of its arguments so they have the same number of dimensions.
    nd = max(a.ndim for a in args)
    return (a[(None, ) * (nd - a.ndim)] for a in args)


def transpose(a, axes=None):
    a = tensor(a)
    if axes is None:
        # By default, transpose reverse the axes
        axes = tuple(range(a.ndim - 1, -1, -1))

    def grad_transpose(leaf_id):
        return a.compute_grad(leaf_id).transpose(
            tuple(range(len(leaves[leaf_id]))) + a.grad_axes(axes))

    return t.Tensor(a.data.transpose(axes), grad_transpose)


def reshape(a, shape):
    a = tensor(a)

    def grad_reshape(leaf_id):
        return a.compute_grad(leaf_id).reshape(leaves[leaf_id] +
                                               np.index_exp[shape])

    return t.Tensor(a.data.reshape(shape), grad_reshape)


def index(a, key):
    def grad_index(leaf_id):
        return a.compute_grad(leaf_id)[(slice(None, None, None), ) *
                                       (len(leaves[leaf_id])) +
                                       np.index_exp[key]]

    return t.Tensor(a.data[key], grad_index)


def add(a, b):
    a, b = tensors(a, b)
    a, b = samedims(a, b)

    def grad_add(leaf_id):
        return a.compute_grad(leaf_id) + b.compute_grad(leaf_id)

    return t.Tensor(a.data + b.data, grad_add)


def neg(a):
    a = tensor(a)

    def grad_neg(leaf_id):
        return -a.compute_grad(leaf_id)

    return t.Tensor(-a.data, grad_neg)


def sub(a, b):
    a, b = tensors(a, b)
    return a + (-b)


def mul(a, b):
    a, b = tensors(a, b)
    a, b = samedims(a, b)

    def grad_mul(leaf_id):
        return a.compute_grad(leaf_id) * b + a * b.compute_grad(leaf_id)

    return t.Tensor(a.data * b.data, grad_mul)


def div(a, b):
    a, b = tensors(a, b)
    a, b = samedims(a, b)

    def grad_truediv(leaf_id):
        return (a.compute_grad(leaf_id) * b - a * b.compute_grad(leaf_id)) / (
            b**2)

    return t.Tensor(a.data / b.data, grad_truediv)


def pow(a, b):
    a, b = tensors(a, b)
    a, b = samedims(a, b)

    def grad_pow(leaf_id):
        if b.dtype == int:
            # Defined even if a < 0
            return b * a.compute_grad(leaf_id) * a**(b - 1)
        # Defined for a > 0
        return (b * a.compute_grad(leaf_id) +
                b.compute_grad(leaf_id) * a * log(a)) * a**(b - 1)

    return t.Tensor(a.data**b.data, grad_pow)


def tensordot(a, b, axes):
    a, b = tensors(a, b)

    def grad_tensordot(leaf_id):
        axes_a, axes_b = axes
        axes_grad_a = a.grad_axes(tuple(axes_a))
        axes_grad_b = b.grad_axes(tuple(axes_b))
        return tensordot(a.compute_grad(leaf_id), b, (axes_grad_a, axes_b)) + \
               tensordot(b.compute_grad(leaf_id), a, (axes_grad_b, axes_a))

    return t.Tensor(np.tensordot(a.data, b.data, axes), grad_tensordot)


def dot(a, b):
    a, b = tensors(a, b)
    if a.ndim == 1 and b.ndim == 1:
        return tensordot(a, b, ([0], [0]))
    elif a.ndim == 2 and b.ndim == 2:
        return a @ b
    elif a.ndim == 0 or b.ndim == 0:
        return a * b
    elif b.ndim == 1:
        return tensordot(a, b, ([-1], [0]))
    else:
        return tensordot(a, b, ([-1], [-2]))


def matmul(a, b):
    a, b = tensors(a, b)
    if a.ndim == 1:
        return matmul(a[None, :], b)[0]
    elif b.ndim == 1:
        return matmul(a, b[:, None])[..., 0]
    a, b = samedims(a, b)

    def grad_matmul(leaf_id):
        return a.compute_grad(leaf_id) @ b + a @ b.compute_grad(leaf_id)

    return t.Tensor(a.data @ b.data, grad_matmul)


def inv(a):
    def grad_inv(leaf_id):
        i = inv(a)  # can't be memoized, this needs to be a t.Tensor
        i = i[(None, ) * len(leaves[leaf_id])]
        return -i @ a.compute_grad(leaf_id) @ i

    return t.Tensor(np.linalg.inv(a.data), grad_inv)


def sum(a, axis=None):
    a = tensor(a)
    if axis is None:
        axis = tuple(range(a.ndim))

    def grad_sum(leaf_id):
        return a.compute_grad(leaf_id).sum(a.grad_axes(axis))

    return t.Tensor(np.sum(a.data, axis), grad_sum)


def exp(a):
    def grad_exp(leaf_id):
        return a.compute_grad(leaf_id) * exp(a)

    return t.Tensor(np.exp(a.data), grad_exp)


def log(a):
    def grad_log(leaf_id):
        return a.compute_grad(leaf_id) / a

    return t.Tensor(np.log(a.data), grad_log)


def zeros(*args, **kwargs):
    return tensor(np.zeros(*args, **kwargs))


def ones(*args, **kwargs):
    return tensor(np.ones(*args, **kwargs))


def empty(*args, **kwargs):
    return tensor(np.empty(*args, **kwargs))


def random(*args, **kwargs):
    return tensor(np.random.random(*args, **kwargs))
