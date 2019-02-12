''' This modules provide a Tensor class, suitable for automatic differentiation
of common numpy functions. '''

import numpy as np

leaves = []  # Array of shapes of leaves


def constant(x):
    ''' A constant is a tensor whose gradient to itself is always zero.
    This is necessary so that x.compute_grad(x.id).compute_grad(x.id) can be zero.
    As a user, you should not use this function (use tensor() instead).
    '''

    def grad_constant(leaf_id):
        return constant(np.zeros(leaves[leaf_id] + x.shape))

    return Tensor(x, grad_constant)


def leaf(x):
    ''' A leaf is a tensor whose gradient to itself is the constant I.
    As a user, you should not use this function (use tensor() instead).
    '''
    x = np.array(x)
    x_id = len(leaves)
    leaves.append(x.shape)

    def grad_leaf(leaf_id):
        if leaf_id == x_id:
            return constant(np.eye(x.size).reshape(x.shape + x.shape))
        else:
            return constant(np.zeros(leaves[leaf_id] + x.shape))

    return Tensor(x, grad_leaf, x_id)


def tensor(x):
    ''' Makes sure x is a tensor. '''
    if isinstance(x, Tensor):
        return x
    else:
        return leaf(x)


class Tensor:
    ''' A Tensor is only a wrapper to an immutable numpy array. '''

    def __init__(self, data, grad_fn, leaf_id=None):
        self.data = np.array(data)
        self.shape = self.data.shape
        self.size = self.data.size
        self.ndim = self.data.ndim
        self.dtype = self.data.dtype

        self.id = leaf_id
        self.grad = {}
        self.grad_fn = grad_fn

    def compute_grad(self, leaf_id):
        ''' tensor.compute_grad(leaf.id) returns the gradient of tensor with respect to leaf.
        The return shape is leaf.shape + tensor.shape.
        '''
        if leaf_id not in self.grad:
            self.grad[leaf_id] = self.grad_fn(leaf_id)
        return self.grad[leaf_id]

    def detach(self):
        ''' Returns a new leaf tensor with the same data. '''
        return leaf(self.data)

    def __str__(self):
        return str(self.data)

    def grad_axes(self, axes):
        ''' Convert axes into self to axes into self.grad. '''
        return tuple(d if d < 0 else d - self.ndim for d in np.index_exp[axes])

    @property
    def T(self):
        return self.transpose()

    def transpose(self, axes=None):
        def grad_transpose(leaf_id):
            n = self.ndim
            ax = axes
            if ax is None:
                # By default, transpose reverse the axes
                ax = tuple(range(n - 1, -1, -1))
            grad = self.compute_grad(leaf_id)
            return grad.transpose(
                tuple(range(grad.ndim - self.ndim)) + self.grad_axes(ax))

        return Tensor(self.data.transpose(axes), grad_transpose)

    def reshape(self, shape):
        def grad_reshape(leaf_id):
            grad = self.compute_grad(leaf_id)
            return grad.reshape(grad.shape[:-self.ndim] + np.index_exp[shape])

        return Tensor(self.data.reshape(shape), grad_reshape)

    def __getitem__(self, key):
        def grad_getitem(leaf_id):
            grad = self.compute_grad(leaf_id)
            return grad[(slice(None, None, None), ) * (grad.ndim - self.ndim) +
                        np.index_exp[key]]

        return Tensor(self.data[key], grad_getitem)

    def __add__(self, other):
        other = tensor(other)

        def grad_add(leaf_id):
            return self.compute_grad(leaf_id) + other.compute_grad(leaf_id)

        return Tensor(self.data + other.data, grad_add)

    def __neg__(self):
        def grad_neg(leaf_id):
            return -self.compute_grad(leaf_id)

        return Tensor(-self.data, grad_neg)

    def __sub__(self, other):
        other = tensor(other)
        return self + (-other)

    def __mul__(self, other):
        other = tensor(other)

        def grad_mul(leaf_id):
            return self.compute_grad(
                leaf_id) * other + self * other.compute_grad(leaf_id)

        return Tensor(self.data * other.data, grad_mul)

    def __truediv__(self, other):
        other = tensor(other)

        def grad_truediv(leaf_id):
            return (self.compute_grad(leaf_id) * other -
                    self * other.compute_grad(leaf_id)) / (other**leaf(2))

        return Tensor(self.data / other.data, grad_truediv)

    def __pow__(self, other):
        other = tensor(other)

        def grad_pow(leaf_id):
            if other.dtype == int:
                # Defined even if self < 0
                return other * self.compute_grad(leaf_id) * self**(
                    other - leaf(1))
            # Defined for self > 0
            return (other * self.compute_grad(leaf_id) +
                    other.compute_grad(leaf_id) * self * self.log()) * self**(
                        other - leaf(1))

        return Tensor(self.data**other.data, grad_pow)

    def tensordot(self, other, axes):
        other = tensor(other)

        def grad_tensordot(leaf_id):
            axes_self, axes_other = axes
            axes_grad_self = self.grad_axes(tuple(axes_self))
            axes_grad_other = other.grad_axes(tuple(axes_other))
            return self.compute_grad(leaf_id).tensordot(other, (axes_grad_self, axes_other)) + \
                   self.tensordot(other.compute_grad(leaf_id), (axes_self, axes_grad_other))

        return Tensor(
            np.tensordot(self.data, other.data, axes), grad_tensordot)

    def dot(self, other):
        other = tensor(other)
        if self.ndim == 1 and other.ndim == 1:
            return self.tensordot(other, ([0], [0]))
        elif self.ndim == 2 and other.ndim == 2:
            return self @ other
        elif self.ndim == 0 or other.ndim == 0:
            return self * other
        elif other.ndim == 1:
            return self.tensordot(other, ([-1], [0]))
        else:
            return self.tensordot(other, ([-1], [-2]))

    def matmul(self, other):
        return self @ other

    def __matmul__(self, other):
        other = tensor(other)

        def grad_matmul(leaf_id):
            return self.compute_grad(
                leaf_id) @ other + self @ other.compute_grad(leaf_id)

        return Tensor(self.data @ other.data, grad_matmul)

    def inv(self):
        def grad_inv(leaf_id):
            inv = self.inv(
            )  # can't be memoized because this needs to be a Tensor
            return -self.compute_grad(leaf_id).tensordot(
                inv, ([-2], [1])).tensordot(inv, ([-2], [0]))

        return Tensor(np.linalg.inv(self.data), grad_inv)

    def sum(self, axis=None):
        def grad_sum(leaf_id):
            ax = axis
            if ax is None:
                ax = tuple(range(self.ndim))
            ax = self.grad_axes(ax)
            return self.compute_grad(leaf_id).sum(ax)

        return Tensor(np.sum(self.data, axis), grad_sum)

    def exp(self):
        def grad_exp(leaf_id):
            return self.compute_grad(leaf_id) * self.exp()

        return Tensor(np.exp(self.data), grad_exp)

    def log(self):
        def grad_log(leaf_id):
            return self.compute_grad(leaf_id) / self

        return Tensor(np.log(self.data), grad_log)


def T(x):
    return tensor(x).T


def transpose(x, axes=None):
    return tensor(x).transpose(axes)


def reshape(x, shape):
    return tensor(x).reshape(shape)


def add(x, y):
    return tensor(x) + y


def neg(x):
    return -tensor(x)


def sub(x, y):
    return tensor(x) - y


def mul(x, y):
    return tensor(x) * y


def div(x, y):
    return tensor(x) / y


def pow(x, y):
    return tensor(x)**y


def tensordot(x, y, axes):
    return tensor(x).tensordot(y, axes)


def dot(x, y):
    return tensor(x).dot(y)


def matmul(x, y):
    return tensor(x).matmul(y)


def inv(x):
    return tensor(x).inv()


def sum(x, axis=None):
    return tensor(x).sum(axis)


def exp(x):
    return tensor(x).exp()


def log(x):
    return tensor(x).log()
