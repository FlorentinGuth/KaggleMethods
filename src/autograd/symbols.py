from . import ops

""" Symbolic calculus, to avoid unnecessary copies.
For instance, this simplifies a + 0 to a and a * 0 to 0.
With hindsight, this should have replaced the closures of autograd.
"""


class Symbol:
    def is_zero(self):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError

    def __add__(self, other):
        return Sum(self, other)

    def __mul__(self, other):
        return Linear(ops.mul, self, other)

    def __sub__(self, other):
        return self + Linear(ops.neg, other)

    def __matmul__(self, other):
        return Linear(ops.matmul, self, other)


class Leaf(Symbol):
    def __init__(self, tensor):
        self.tensor = tensor

    def is_zero(self):
        return False

    def compute(self):
        return self.tensor


class Grad(Symbol):
    def __init__(self, tensor, leaf_id):
        self.tensor = tensor
        self.leaf_id = leaf_id

    def is_zero(self):
        return self.leaf_id not in self.tensor.children_ids

    def compute(self):
        return self.tensor.compute_grad(self.leaf_id)


class Sum(Symbol):
    def __init__(self, *symbols):
        self.symbols = symbols

    def is_zero(self):
        return all(symbol.is_zero() for symbol in self.symbols)

    def compute(self):
        res = None
        for symbol in self.symbols:
            if not symbol.is_zero():
                if res is None:
                    res = symbol.compute()
                else:
                    res = res + symbol.compute()
        return res


class Linear(Symbol):
    def __init__(self, op, *symbols):
        self.symbols = symbols
        self.op = op

    def is_zero(self):
        return any(symbol.is_zero() for symbol in self.symbols)

    def compute(self):
        return self.op(*(symbol.compute() for symbol in self.symbols))


class Tensordot(Linear):
    def __init__(self, symbol1, symbol2, axes):
        super().__init__(lambda a, b: ops.tensordot(a, b, axes), symbol1, symbol2)
