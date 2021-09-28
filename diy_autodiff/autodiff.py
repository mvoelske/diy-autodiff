#%%
import math
from abc import abstractmethod

import tqdm # type: ignore

class BaseTerm:
    def __init__(self, value):
        self.type = ''
        self.value = value

    def compute(self):
        return self.value

    def __repr__(self):
        return f"{self.value}{self.type}"

    def __add__(self, o):
        return Addition(self, o)

    def __mul__(self, o):
        return Multiplication(self, o)

    def __sub__(self, o):
        return Addition(self, Negation(o))

class Variable(BaseTerm):
    def __init__(self, value):
        super().__init__(value)
        self.type = 'v'
    
    def update(self, value):
        self.value = value

class Constant(BaseTerm):
    def __init__(self, value):
        super().__init__(value)
        self.type = 'c'

########################################################################

class BaseOperation(BaseTerm):
    """Abstract base class for an operation with an arbitrary number of
    operands."""
    @abstractmethod
    def compute(self) -> float:
        """Evaluates this operation with respect to its operand(s)

        Returns:
            float: the result of applying this operation to its operand(s)
        """
        ...

    @abstractmethod
    def derive(self) -> tuple[float]:
        """Computes the gradient of this operation with respect to its
        operand(s).

        Returns:
            tuple[float]: length corresponds to the number of operands, and the
                i-th element is the gradient with respect to the i-th operand
        """
        ...

    @abstractmethod
    def operands(self) -> tuple[BaseTerm]:
        """Return the operand(s) in order.

        Returns:
            tuple[BaseTerm]: this operation's operand(s)
        """
        ...

class UnaryOperation(BaseOperation):
    """An operation with a single operand."""
    def __init__(self, a: BaseTerm):
        self.a = a
    def __repr__(self):
        return f'{self.__class__.__name__}({self.a})'
    def operands(self):
        return (self.a,)

class BinaryOperation(BaseOperation):
    """An operation with exactly two operands."""
    def __init__(self, a: BaseTerm, b: BaseTerm):
        self.a = a
        self.b = b
    def __repr__(self):
        return f'{self.__class__.__name__}({self.a}, {self.b})'
    def operands(self):
        return (self.a, self.b)

class Negation(UnaryOperation):
    def compute(self):
        return -self.a.compute()
    def derive(self):
        return (-1,)

class Addition(BinaryOperation):
    def compute(self):
        return self.a.compute() + self.b.compute()
    def derive(self):
        return (1, 1)

class Multiplication(BinaryOperation):
    def compute(self):
        return self.a.compute() * self.b.compute()
    def derive(self):
        return (self.b.compute(), self.a.compute())

class Sigmoid(UnaryOperation):
    @staticmethod
    def _s(z):
        return 1/(1 + math.exp(- z))
    def compute(self):
        return Sigmoid._s(self.a.compute())
    def derive(self):
        a = self.a.compute()
        return (Sigmoid._s(a) * (1 - Sigmoid._s(a)),)

class Square(UnaryOperation):
    def compute(self):
        return self.a.compute() ** 2
    def derive(self):
        return (2 * self.a.compute(),)

########################################################################

def grad(f: BaseOperation, vars: list[Variable]) -> dict[Variable, float]:
    """Perform reverse-mode automatic differentiation of f with respect to vars
    at their current values.

    Args:
        f (BaseOperation): The function to differentiate
        vars (list[Variable]): The point where the function is to be differentiated

    Returns:
        dict[Variable, float]: mapping from variables to their gradients
    """
    grads = {v:0.0 for v in vars}

    ops, derivs = f.operands(), f.derive()
    
    for op, d in zip(ops, derivs):
        if op in grads:
            grads[op] += d # type: ignore
        elif isinstance(op, BaseOperation):
            op_grads = grad(op, vars)
            grads = {v:(grads[v] + d*op_grads[v]) for v in grads}

    return grads

########################################################################

def gradient_descent(
    training_loss: UnaryOperation,
    params: list[Variable],
    eta: float, iterations: int,
    validation_loss: UnaryOperation = None, val_every: int = None):

    pb = tqdm.tqdm(range(iterations), desc="GD", leave=True)

    for i in pb:
        g = grad(training_loss, params)
        for p in params:
            p.update(p.value - eta * g[p])
        if validation_loss is not None and i % val_every == 0:
            pb.set_postfix({'val.loss': validation_loss.compute()})

# %%
