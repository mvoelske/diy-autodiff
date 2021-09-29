import math
from abc import abstractmethod

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

    def __neg__(self):
        return Negation(self)

    def __truediv__(self, o):
        return self * Reciprocal(o)

    def __pow__(self, p):
        if p == 2:
            return Square(self)
        else:
            raise RuntimeError("powers other than 2 not yet implemented")

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
    def derive(self) -> tuple[float, ...]:
        """Computes the gradient of this operation with respect to its
        operand(s).

        Returns:
            tuple[float, ...]: length corresponds to the number of operands, and the
                i-th element is the gradient with respect to the i-th operand
        """
        ...

    @abstractmethod
    def operands(self) -> tuple[BaseTerm, ...]:
        """Return the operand(s) in order.

        Returns:
            tuple[BaseTerm, ...]: this operation's operand(s)
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

class Identity(UnaryOperation):
    def compute(self):
        return self.a.compute()
    def derive(self):
        return (1, )

class Negation(UnaryOperation):
    def compute(self):
        return -self.a.compute()
    def derive(self):
        return (-1,)

class Log(UnaryOperation):
    def compute(self):
        return math.log(self.a.compute())
    def derive(self):
        return (1 / self.a.compute(),)

class Exp(UnaryOperation):
    def compute(self):
        return math.exp(self.a.compute())
    def derive(self):
        return (self.compute(),)

class Reciprocal(UnaryOperation):
    def compute(self):
        return 1 / self.a.compute()
    def derive(self):
        a = self.a.compute()
        if a == 0:
            return (0,)
        return (- a**(-2),)

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

class ReLU(UnaryOperation):
    def compute(self):
        return max(0, self.a.compute())
    def derive(self):
        return (1 if self.a.compute() >= 0 else 0,)

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
    assert len(ops) == len(derivs)
    
    for op, d in zip(ops, derivs):
        if op in grads:
            grads[op] += d # type: ignore
        elif isinstance(op, BaseOperation):
            op_grads = grad(op, vars)
            grads = {v:(grads[v] + d*op_grads[v]) for v in grads}

    return grads

########################################################################