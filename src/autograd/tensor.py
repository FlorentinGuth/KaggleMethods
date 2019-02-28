from . import ops
import numpy as np


class Tensor:
    """ A Tensor is only a wrapper to an immutable numpy array. """

    def __init__(self, data, grad_fn, leaf_id=None):
        self.data = np.array(data, copy=False)
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
            if self.grad[leaf_id].shape != ops.leaves[leaf_id] + self.shape:
                raise ValueError('shape mismatch: gradient of {} wrt {} is {} (culprit is {})'
                                 .format(self.shape, ops.leaves[leaf_id],
                                         self.grad[leaf_id].shape, self.grad_fn.__name__))
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
        res = tuple(d if d < 0 else d - self.ndim for d in np.index_exp[axes])
        if isinstance(axes, int):
            return res[0]
        return res

    @property
    def T(self):
        return self.transpose()

    def transpose(self, axes=None):
        return ops.transpose(self, axes)

    def moveaxis(self, source, destination):
        return ops.moveaxis(self, source, destination)

    def reshape(self, shape):
        return ops.reshape(self, shape)

    def __getitem__(self, key):
        return ops.index(self, key)

    def __add__(self, other):
        return ops.add(self, other)

    def __radd__(self, other):
        return ops.add(other, self)

    def __neg__(self):
        return ops.neg(self)

    def __sub__(self, other):
        return ops.sub(self, other)

    def __rsub__(self, other):
        return ops.sub(other, self)

    def __mul__(self, other):
        return ops.mul(self, other)

    def __rmul__(self, other):
        return ops.mul(other, self)

    def __truediv__(self, other):
        return ops.div(self, other)

    def __rtruediv__(self, other):
        return ops.div(other, self)

    def __pow__(self, other):
        return ops.pow(self, other)

    def __rpow__(self, other):
        return ops.pow(other, self)

    def tensordot(self, other, axes):
        return ops.tensordot(self, other, axes)

    def dot(self, other):
        return ops.dot(self, other)

    def matmul(self, other):
        return self @ other

    def __matmul__(self, other):
        return ops.matmul(self, other)

    def __rmatmul__(self, other):
        return ops.matmul(other, self)

    def inv(self):
        return ops.inv(self)

    def sum(self, axis=None):
        return ops.sum(self, axis)

    def mean(self, axis=None):
        return ops.mean(self, axis)

    def exp(self):
        return ops.exp(self)

    def log(self):
        return ops.log(self)
