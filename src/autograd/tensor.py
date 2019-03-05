from . import ops
import numpy as np


config = {
    'debug': False,  # Whether to do some checks / prints.
    'grad':  True,   # Whether to store gradient closures.
}


class Config:
    """ Context to change autograd global config.
    Usage:
    >>> with ag.Config(grad=False):
    >>>     some_computation()
    """
    def __init__(self, **kwargs):
        self.updates = kwargs

    def __enter__(self):
        self.old_config = config.copy()
        config.update(self.updates)

    def __exit__(self, exc_type, exc_val, exc_tb):
        global config
        config = self.old_config


class Tensor:
    """ A Tensor is only a wrapper to an immutable numpy array. """

    def __init__(self, data, grad_fn, leaf_id=None, requires_grad=False, children=None):
        """ Creates a new tensor with the given data and grad function.
        Two typical use cases:
        - To create a leaf:
        >>> Tensor(data, grad_fn, leaf_id, requires_grad)
        - To create a new node:
        >>> Tensor(data, grad_fn, children=children)
        Basically, you should either supply leaf_id and require_grad or children, as a node with children is not a leaf
        (it does not make much sense to differentiate with respect to it) and it requires gradient only if one of its
        children requires it.

        :param data: A numpy array containing the data of the tensor. To the user, a tensor is more or less its data.
        :param grad_fn: A function from a leaf_id to a tensor, the gradient of data with respect to the leaf.
        :param leaf_id: A unique id if the tensor is a leaf.
        :param requires_grad: If False, then we know we will not compute gradients with respect to self, so it can be
        assumed constant everywhere and it frees memory by not storing the gradient closures.
        :param children: If the tensor is not a leaf, an iterable of the tensors it directly depends on.
        """
        self.data = np.array(data, copy=False)
        self.shape = self.data.shape
        self.strides = self.data.strides
        self.size = self.data.size
        self.ndim = self.data.ndim
        self.dtype = self.data.dtype

        if children is not None:
            self.requires_grad = any(child.requires_grad for child in children)
            self.children_ids = frozenset().union(*(child.children_ids for child in children))
        else:
            self.requires_grad = requires_grad
            self.children_ids = frozenset([leaf_id])

        self.id = leaf_id
        self.grad = dict()
        self.grad_fn = grad_fn if config['grad'] and self.requires_grad else None

    def compute_grad(self, leaf_id):
        """ tensor.compute_grad(leaf.id) returns the gradient of tensor with respect to leaf.
        The return shape is broadcastable to leaf.shape + tensor.shape (with the same number of dimensions).
        """

        if leaf_id not in self.grad:
            if leaf_id in self.children_ids:
                self.grad[leaf_id] = self.grad_fn(leaf_id)
            else:
                # self.grad[leaf_id] = ops.leaf(np.zeros(())[(None,) * (ops.leaf_ndim(leaf_id) + self.ndim)])
                self.grad[leaf_id] = ops.leaf(np.zeros(ops.leaf_shape(leaf_id) + self.shape))

            if config['debug']:
                if self.grad[leaf_id].shape != self.grad_shape(leaf_id):
                    raise ValueError('shape mismatch: gradient of {} wrt {} is {} (culprit is {})'
                                     .format(self.shape, ops.leaf_shape(leaf_id),
                                             self.grad[leaf_id].shape, self.grad_fn.__name__))

                if self.grad[leaf_id].data.max() != self.grad[leaf_id].data.max():
                    raise ValueError('NaNs in gradient (culprit is {}'.format(self.grad_fn.__name__))

        return self.grad[leaf_id]

    def detach(self, requires_grad=None):
        """ Returns a new leaf tensor with the same data.
        By default, requires_grad is self.requires_grad.
        """
        if requires_grad is None:
            requires_grad = self.requires_grad
        return ops.leaf(self.data, requires_grad)

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return 'tensor({})'.format(repr(self.data))

    def grad_axes(self, axes):
        """ Convert axes into self to axes into self.grad. """
        if isinstance(axes, list):
            axes = tuple(axes)

        res = tuple(d if d < 0 else d - self.ndim for d in np.index_exp[axes])

        if isinstance(axes, int):
            return res[0]
        return res

    def grad_shape(self, leaf_id):
        """ Returns the shape of the gradient (or the shape it can broadcast to). """
        return ops.leaf_shape(leaf_id) + self.shape

    @property
    def T(self):
        return self.transpose()

    def transpose(self, axes=None):
        return ops.transpose(self, axes)

    def moveaxis(self, source, destination):
        return ops.moveaxis(self, source, destination)

    def reshape(self, shape):
        return ops.reshape(self, shape)

    def expand(self, axes, shape):
        return ops.expand(self, axes, shape)

    def broadcast_to(self, shape):
        return ops.broadcast_to(self, shape)

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
