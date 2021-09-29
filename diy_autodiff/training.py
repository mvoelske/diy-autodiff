from abc import abstractmethod
from typing import Callable

import tqdm # type: ignore

from . import autodiff
from .autodiff import BaseOperation, BaseTerm, Constant, BaseTerm, Identity, grad 
from .models import Model

class LossFunction:
    @abstractmethod
    def __call__(self, c: list[Constant], y: list[BaseTerm]) -> BaseOperation:
        ...

class SquaredLoss(LossFunction):
    def __call__(self, c: list[Constant], y: list[BaseTerm]) -> BaseOperation:
        l = Identity(Constant(0))
        for ci, yi in zip(c, y):
            l += (ci - yi)**2
        return l

class SoftmaxCrossEntropyLoss(LossFunction):
    def __call__(self, c: list[Constant], y: list[BaseTerm]) -> BaseOperation:
        import math
        # an optimisation to avoid gratuitous gradient passes through here
        log_sum = Constant(math.log(sum((math.exp(yi.compute()) for yi in y))))
        return -sum((ci * (yi - log_sum) for ci, yi in zip(c, y)), Constant(0))

########################################################################

def batch_gradient_descent(
    model: Model,
    loss: LossFunction,
    training_set: list[tuple[list[Constant], list[Constant]]],
    iterations: int, eta: float,
    batch_size: int = None,
    validation_set: list[tuple[list[Constant], list[Constant]]] = None,
    validate_every: int = 1, val_cb: Callable[[], None] = None):

    pb = tqdm.tqdm(range(iterations), desc='Epoch', leave=True)

    params = model.parameters()
    initial_loss = Identity(Constant(0))
    if batch_size is None:
        batch_size = len(training_set)

    for i in pb:
        epoch_loss = initial_loss
        batches = (
            training_set[i:batch_size+i]
            for i in range(0, len(training_set), batch_size))
        for b in tqdm.tqdm(batches, desc='Batch', leave=False):
            batch_loss = initial_loss
            for x, c in b:
                y = model(x)
                batch_loss += loss(c, y)
        
            g = grad(batch_loss, params)
            epoch_loss += batch_loss
            for p in params:
                p.update(p.value - eta * g[p])

        if i % validate_every == 0:
            info = {'train.loss': epoch_loss.compute()}

            if validation_set is not None:
                val_loss = initial_loss
                for x, c in tqdm.tqdm(validation_set, desc="Validate", leave=False):
                    y = model(x)
                    val_loss += loss(c, y)
                info['val.loss'] = val_loss.compute()
            if val_cb is not None:
                val_cb()
            pb.set_postfix(info)

########################################################################

def as_dataset(xs: list[list[float]], cs: list[list[float]]) -> list[tuple[list[Constant], list[Constant]]]:
    assert len(xs) == len(cs)
    ds = []
    for x, c in zip(xs, cs):
        ds.append(([Constant(xi) for xi in x], [Constant(ci) for ci in c]))
    return ds