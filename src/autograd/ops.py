from . import tensor as t
import numpy as np

leaves = []  # Array of shapes of leaves


def leaf_shape(leaf_id):
    return leaves[leaf_id]


def leaf_ndim(leaf_id):
    return len(leaves[leaf_id])


def leaf(x, requires_grad=False):
    """ A leaf is a tensor whose gradient to itself is the constant I.
    As a user, you should not use this function (use tensor() instead).
    """
    x = np.array(x, copy=False)
    x_id = len(leaves)
    leaves.append(x.shape)

    def grad_leaf(leaf_id):
        # leaf_id = x_id, otherwise the optimization would have pruned that part of the graph.
        assert leaf_id == x_id
        return leaf(np.eye(x.size).reshape(x.shape + x.shape))

    return t.Tensor(x, grad_leaf, x_id, requires_grad)


def tensor(x, requires_grad=False):
    """ Makes sure x is a tensor. """
    if isinstance(x, t.Tensor):
        return x
    else:
        return leaf(x, requires_grad)


def tensor_aggregate(x, requires_grad=False):
    """ Makes sure x is a tensor, but works properly with lists/tuples of tensors. 
    Typical use case is sum(list_of_tensors_with_grad).
    """
    if isinstance(x, np.ndarray):
        return leaf(x, requires_grad=False)
    elif not isinstance(x, t.Tensor):
        # tuple, list, iterable...
        return stack(x)
    else:
        return x


def tensors(*args):
    return tuple(tensor(a) for a in args)


def samedims(*args):
    """ Prepend 1s to the shape of its arguments so they have the same number of dimensions. """
    nd = max(a.ndim for a in args)
    if nd == min(a.ndim for a in args):
        # Simple optimization to avoid adding useless nodes to the graph
        return args

    return tuple(a[(None, ) * (nd - a.ndim)] for a in args)


def broadcastable(*shapes):
    """ Returns True if the given shapes are broadcastable to each other. """
    nd = max(len(shape) for shape in shapes)
    shapes = [(1,)*(nd - len(shape)) + shape for shape in shapes]
    full_shape = tuple(max(shape[i] for shape in shapes) for i in range(nd))
    return all(shape[i] == 1 or shape[i] == full_shape[i] for i in range(nd) for shape in shapes)


def transpose(a, axes=None):
    a = tensor(a)
    if axes is None:
        # By default, transpose reverse the axes
        axes = tuple(range(a.ndim - 1, -1, -1))

    def grad_transpose(leaf_id):
        return a.compute_grad(leaf_id).transpose(
            tuple(range(leaf_ndim(leaf_id))) + a.grad_axes(axes))

    return t.Tensor(a.data.transpose(axes), grad_transpose, children=[a])


def moveaxis(a, source, destination):
    a = tensor(a)

    def grad_moveaxis(leaf_id):
        return a.compute_grad(leaf_id).moveaxis(a.grad_axes(source), a.grad_axes(destination))

    return t.Tensor(np.moveaxis(a.data, source, destination), grad_moveaxis, children=[a])


def reshape(a, shape):
    a = tensor(a)

    def grad_reshape(leaf_id):
        grad = a.compute_grad(leaf_id)
        return grad.reshape(grad.shape[:leaf_ndim(leaf_id)] + np.index_exp[shape])

    return t.Tensor(a.data.reshape(shape), grad_reshape, children=[a])


def expand(a, axes, shape):
    """ Returns a view of a such that for each i, a.shape[axes[i]] = shape[i]. """
    a = tensor(a)

    full_shape = list(a.shape)
    full_strides = list(a.strides)
    need_change = False
    for i in range(len(axes)):
        ax = axes[i]
        if full_shape[ax] != shape[i]:
            assert full_shape[ax] == 1
            full_shape[ax] = shape[i]
            full_strides[ax] = 0
            need_change = True

    if not need_change:
        # Simple optimization to avoid adding useless nodes to the graph
        return a

    def grad_expand(leaf_id):
        return expand(a.compute_grad(leaf_id), a.grad_axes(axes), shape)

    return t.Tensor(np.lib.stride_tricks.as_strided(a.data, full_shape, full_strides), grad_expand, children=[a])


def expand_arrays(arrays, axes):
    """ Returns views of arrays such that arrays[i].shape[*axes[i]] is constant. """
    arrays = tensors(*arrays)

    n, m = len(axes), len(axes[0])
    shape = [1] * m
    for i in range(n):
        for j in range(m):
            shape[j] = max(shape[j], arrays[i].shape[axes[i][j]])

    return tuple(expand(arrays[i], axes[i], shape) for i in range(n))


def broadcast_to(a, shape):
    a = tensor(a)
    if a.shape == shape:
        return a

    def grad_broadcast_to(leaf_id):
        grad = a.compute_grad(leaf_id)
        return broadcast_to(grad, grad.shape[:leaf_ndim(leaf_id)] + shape)

    return t.Tensor(np.broadcast_to(a.data, shape), grad_broadcast_to, children=[a])


def broadcast_arrays(*arrays):
    arrays = tensors(*arrays)
    arrays = samedims(*arrays)

    def grad_broadcast_arrays(leaf_id):
        return broadcast_arrays(*(a.compute_grad(leaf_id) for a in arrays), grad_broadcast_arrays)

    data = np.broadcast_arrays(*(a.data for a in arrays))
    return tuple(t.Tensor(data[i], grad_broadcast_arrays, children=[arrays[i]]) for i in range(len(arrays)))


def index(a, key):
    a = tensor(a)

    def grad_index(leaf_id):
        return a.compute_grad(leaf_id)[(slice(None, None, None), ) * leaf_ndim(leaf_id)
                                                       + np.index_exp[key]]

    return t.Tensor(a.data[key], grad_index, children=[a])


def concatenate(arrays, axis=0):
    arrays = tensors(*arrays)
    arrays = samedims(*arrays)
    axes = list(range(axis)) + list(range(axis + 1, arrays[0].ndim))
    arrays = expand_arrays(arrays, [axes for _ in range(len(arrays))])

    def grad_concatenate(leaf_id):
        return concatenate(tuple(a.compute_grad(leaf_id) for a in arrays), arrays[0].grad_axes(axis))

    return t.Tensor(np.concatenate(tuple(a.data for a in arrays), axis), grad_concatenate, children=arrays)


def stack(arrays, axis=0):
    arrays = tensors(*arrays)
    arrays = samedims(*arrays)
    arrays = expand_arrays(arrays, [list(range(arrays[0].ndim)) for _ in range(len(arrays))])

    def grad_stack(leaf_id):
        return stack(tuple(a.compute_grad(leaf_id) for a in arrays),
                     axis if axis < 0 else axis + leaf_ndim(leaf_id))

    return t.Tensor(np.stack(tuple(a.data for a in arrays), axis), grad_stack, children=arrays)


def add(a, b):
    a, b = tensors(a, b)
    a, b = samedims(a, b)

    def grad_add(leaf_id):
        return a.compute_grad(leaf_id) + b.compute_grad(leaf_id)

    return t.Tensor(a.data + b.data, grad_add, children=[a, b])


def neg(a):
    a = tensor(a)

    def grad_neg(leaf_id):
        return -a.compute_grad(leaf_id)

    return t.Tensor(-a.data, grad_neg, children=[a])


def sub(a, b):
    a, b = tensors(a, b)
    return a + (-b)


def mul(a, b):
    a, b = tensors(a, b)
    a, b = samedims(a, b)

    def grad_mul(leaf_id):
        return a.compute_grad(leaf_id) * b + a * b.compute_grad(leaf_id)

    return t.Tensor(a.data * b.data, grad_mul, children=[a, b])


def div(a, b):
    a, b = tensors(a, b)
    a, b = samedims(a, b)

    def grad_truediv(leaf_id):
        return (a.compute_grad(leaf_id) * b - a * b.compute_grad(leaf_id)) / (
            b**2)

    return t.Tensor(a.data / b.data, grad_truediv, children=[a, b])


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

    return t.Tensor(a.data**b.data, grad_pow, children=[a, b])


def tensordot(a, b, axes):
    a, b = tensors(a, b)

    def grad_tensordot(leaf_id):
        axes_a, axes_b = axes
        axes_grad_a = a.grad_axes(tuple(axes_a))
        axes_grad_b = b.grad_axes(tuple(axes_b))
        return (tensordot(a.compute_grad(leaf_id), b, (axes_grad_a, axes_b)) if leaf_id in a.children_ids else 0) + \
               (tensordot(a, b.compute_grad(leaf_id), (axes_a, axes_grad_b))\
                   .moveaxis(tuple(i + a.ndim - len(axes_a) for i in range(leaf_ndim(leaf_id))),
                             tuple(range(leaf_ndim(leaf_id)))) if leaf_id in b.children_ids else 0)

    return t.Tensor(np.tensordot(a.data, b.data, axes), grad_tensordot, children=[a, b])


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
        return (a.compute_grad(leaf_id) @ b if leaf_id in a.children_ids else 0) + \
               (a @ b.compute_grad(leaf_id) if leaf_id in b.children_ids else 0)

    return t.Tensor(a.data @ b.data, grad_matmul, children=[a, b])


def inv(a):
    idata = np.linalg.inv(a.data)

    def grad_inv(leaf_id):
        i = t.Tensor(idata, grad_inv, children=[a])
        return -i @ a.compute_grad(leaf_id) @ i

    return t.Tensor(idata, grad_inv, children=[a])


def sum(a, axis=None):
    a = tensor_aggregate(a)
    if axis is None:
        axis = tuple(range(a.ndim))

    def grad_sum(leaf_id):
        return sum(a.compute_grad(leaf_id), a.grad_axes(axis))

    return t.Tensor(np.sum(a.data, axis), grad_sum, children=[a])


def mean(a, axis=None):
    a = tensor_aggregate(a)
    if axis is None:
        axis = tuple(range(a.ndim))

    def grad_mean(leaf_id):
        return mean(a.compute_grad(leaf_id), a.grad_axes(axis))

    return t.Tensor(np.mean(a.data, axis), grad_mean, children=[a])


def exp(a):
    def grad_exp(leaf_id):
        return a.compute_grad(leaf_id) * exp(a)

    return t.Tensor(np.exp(a.data), grad_exp, children=[a])


def log(a):
    def grad_log(leaf_id):
        return a.compute_grad(leaf_id) / a

    return t.Tensor(np.log(a.data), grad_log, children=[a])


def zeros(*args, requires_grad=False, **kwargs):
    kwargs.setdefault("dtype", np.float32)
    return tensor(np.zeros(*args, **kwargs), requires_grad)


def ones(*args, requires_grad=False, **kwargs):
    kwargs.setdefault("dtype", np.float32)
    return tensor(np.ones(*args, **kwargs), requires_grad)


def empty(*args, requires_grad=False, **kwargs):
    kwargs.setdefault("dtype", np.float32)
    return tensor(np.empty(*args, **kwargs), requires_grad)


def full(*args, requires_grad=False, **kwargs):
    kwargs.setdefault("dtype", np.float32)
    return tensor(np.full(*args, **kwargs), requires_grad)


def eye(*args, requires_grad=False, **kwargs):
    kwargs.setdefault("dtype", np.float32)
    return tensor(np.eye(*args, **kwargs), requires_grad)


def random(*args, requires_grad=False, **kwargs):
    kwargs.setdefault("dtype", np.float32)
    dtype = kwargs["dtype"]
    del kwargs["dtype"]
    return tensor(np.random.random(*args, **kwargs).astype(dtype), requires_grad)
