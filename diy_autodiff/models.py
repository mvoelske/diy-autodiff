from abc import abstractmethod
import random
from typing import Union

from .autodiff import Variable, BaseTerm, UnaryOperation, Identity, Sigmoid


class Model:
    @abstractmethod
    def parameters(self) -> list[Variable]: ...

    @abstractmethod
    def __call__(self, x: list[BaseTerm]) -> list[BaseTerm]: ...

    @abstractmethod
    def initialize(self, seed: int=1): ...



class Perceptron(Model):
    def __init__(self, in_dim: int, out_dim: int,
                 activation: type[UnaryOperation] = Identity):
        self._w = [
            [Variable(None) for _ in range(in_dim+1)]
            for _ in range(out_dim)]
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation
    
    def parameters(self):
        return [p for row in self._w for p in row]

    def __call__(self, x: list[BaseTerm]):
        assert len(x) == self.in_dim
        out = []
        for row in self._w:
            wtx = row[0] # bias
            for xi, wi in zip(x, row[1:]):
                wtx += xi*wi
            out.append(self.activation(wtx))
        return out
    
    def initialize(self, seed=1):
        r = random.Random(seed)
        for row in self._w:
            for w in row:
                w.update(r.normalvariate(0, 1))


class MultiLayerPerceptron(Model):
    def __init__(
        self, in_dim: int, hidden_dims: list[int], out_dim: int,
        activations: Union[list[type[UnaryOperation]], type[UnaryOperation]] = Sigmoid):

        if not isinstance(activations, list):
            activations = [activations for _ in range(len(hidden_dims) + 1)]
        
        assert len(activations) == len(hidden_dims) + 1

        self.layers : list[Model] = []
        self.in_dim = in_dim
        self.out_dim = out_dim

        current = self.in_dim
        for n, act in zip(hidden_dims + [out_dim], activations):
            l = Perceptron(current, n, act)
            self.layers.append(l)
            current = n

    def parameters(self) -> list[Variable]:
        return [l for layer in self.layers for l in layer.parameters()]
    
    def __call__(self, x: list[BaseTerm]):
        y = self.layers[0](x)
        for l in self.layers[1:]:
            y = l(y)
        return y
    
    def initialize(self, seed: int = 1):
        seeder = random.Random(seed)
        for l in self.layers:
            l.initialize(seeder.randint(0, 2**32))
