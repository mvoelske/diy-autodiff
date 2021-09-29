from abc import abstractmethod
from typing import Callable

import tqdm # type: ignore

from .autodiff import BaseOperation, BaseTerm, Constant, BaseTerm, Identity, grad 
from .models import Model

class LossFunction:
    @abstractmethod
    def __call__(self, c: list[Constant], y: list[BaseTerm]) -> BaseOperation:
        ...

class RSSLoss(LossFunction):
    def __call__(self, c: list[Constant], y: list[BaseTerm]) -> BaseOperation:
        l = Identity(Constant(0))
        for ci, yi in zip(c, y):
            l += (ci - yi)**2
        return l

########################################################################

def batch_gradient_descent(
    model: Model,
    loss: LossFunction,
    training_set: list[tuple[list[Constant], list[Constant]]],
    iterations: int, eta: float,
    batch_size: int = None,
    validation_set: list[tuple[list[Constant], list[Constant]]] = None,
    validate_every: int = 1):

    pb = tqdm.tqdm(range(iterations), desc='Epoch', leave=True)

    params = model.parameters()
    initial_loss = Identity(Constant(0))
    if batch_size is None:
        batch_size = len(training_set)

    for i in pb:
        batches = (
            training_set[i:batch_size+i]
            for i in range(0, len(training_set), batch_size))
        for b in tqdm.tqdm(batches, desc='Batch', leave=False):
            batch_loss = initial_loss
            for x, c in b:
                y = model(x)
                batch_loss += loss(c, y)
        
            g = grad(batch_loss, params)
            for p in params:
                p.update(p.value - eta * g[p])

        if validation_set is not None and i % validate_every == 0:
            val_loss = initial_loss
            for x, c in tqdm.tqdm(validation_set, desc="Validate", leave=False):
                y = model(x)
                val_loss += loss(c, y)
            pb.set_postfix({'val.loss': val_loss.compute()})

########################################################################

def as_training_set(xs: list[list[float]], cs: list[list[float]]) -> list[tuple[list[Constant], list[Constant]]]:
    assert len(xs) == len(cs)
    ds = []
    for x, c in zip(xs, cs):
        ds.append(([Constant(xi) for xi in x], [Constant(ci) for ci in c]))
    return ds