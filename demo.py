#!/usr/bin/env python3

import diy_autodiff.autodiff as ad
from diy_autodiff import models, training

from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
import numpy as np

iris = load_iris()

xs = iris['data']
cs = iris['target']

# 1-vs-rest for simplicity
cs = (cs == 0).astype(int)
target_names = '+'.join(iris['target_names'][[1,2]]), iris['target_names'][0]

n_test = len(xs) // 9

train_idx = np.arange(len(xs))
np.random.seed(3)
np.random.shuffle(train_idx)
train_idx, test_idx = train_idx[:len(xs)-n_test], train_idx[-n_test:]

xtrain, ctrain = xs[train_idx,:], cs[train_idx]
xtest, ctest = xs[test_idx,:], cs[test_idx]

ctrain1 = ctrain[:, None]
ctest1 = ctest[:, None]

train = training.as_dataset(
    xtrain, ctrain1
)
test = training.as_dataset(
    xtest, ctest1
)

## define the model to be trained

model = models.MultiLayerPerceptron(
    len(xtrain[0]), [3], len(ctrain1[0]),
    activations=[ad.ReLU, ad.Sigmoid]
)

loss = training.SquaredLoss()

model.initialize(1)

## to demonstrate that something actually happens, show performance of randomized weights first
print(72*'*')
print('Before Training')
print('===============')

def print_eval(dataset, label):
    ys = [[yi.compute() for yi in model(x)] for x, _ in dataset]
    ys = np.array(ys) > 0.5
    cs = [[ci.value for ci in c ] for _, c in dataset]
    print(label)
    print('-'*len(label))
    print(classification_report(cs, ys,
          zero_division=0,
          target_names=target_names))

print_eval(train, 'training set')
print_eval(test, 'test set')

# train model
training.batch_gradient_descent(
    model, loss, train,
    iterations=200, eta=0.02, batch_size=20,
    validation_set=test, validate_every=5,
)

print(72*'*')
print('After Training')
print('==============')

print_eval(train, 'training set')
print_eval(test, 'test set')