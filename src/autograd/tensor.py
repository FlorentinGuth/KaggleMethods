from . import ops
import numpy as np


class Tensor:
    """ A Tensor is only a wrapper to an immutable numpy array. """

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
        """ tensor.compute_grad(leaf.id) returns the gradient of tensor with respect to leaf.
        The return shape is leaf.shape + tensor.shape.
        """
        if leaf_id not in self.grad:
            self.grad[leaf_id] = self.grad_fn(leaf_id)
        return self.grad[leaf_id]

    def detach(self):
        """ Returns a new leaf tensor with the same data. """
        return ops.leaf(self.data)

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return 'tensor({})'.format(repr(self.data))

    def grad_axes(self, axes):
        """ Convert axes into self to axes into self.grad. """
        return tuple(d if d < 0 else d - self.ndim for d in np.index_exp[axes])

    @property
    def T(self):
        return self.transpose()

    def transpose(self, axes=None):
        return ops.transpose(self, axes)

    def reshape(self, shape):
        return ops.reshape(self, shape)

    def __getitem__(self, key):
        return ops.index(self, key)

    def __add__(self, other):
        return ops.add(self, other)

    def __neg__(self):
        return ops.neg(self)

    def __sub__(self, other):
        return ops.sub(self, other)

    def __mul__(self, other):
        return ops.mul(self, other)

    def __truediv__(self, other):
        return ops.div(self, other)

    def __pow__(self, other):
        return ops.pow(self, other)

    def tensordot(self, other, axes):
        return ops.tensordot(self, other, axes)

    def dot(self, other):
        return ops.dot(self, other)

    def matmul(self, other):
        return self @ other

    def __matmul__(self, other):
        return ops.matmul(self, other)

    def inv(self):
        return ops.inv(self)

    def sum(self, axis=None):
        return ops.sum(self, axis)

    def exp(self):
        return ops.exp(self)

    def log(self):
        return ops.log(self)
