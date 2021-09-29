import pytest
from diy_autodiff import __version__
from diy_autodiff import autodiff as ad
from diy_autodiff import training as tr
from diy_autodiff import models as md


def test_version():
    assert __version__ == '0.1.0'


def test_simple_eqn():
    var1 = ad.Variable(1.5)
    var2 = ad.Variable(2.5)
    var3 = ad.Variable(3.5)

    result = var1 + var2*var3 + var1*var2

    assert result.compute() == 14
    d = ad.grad(result, [var1, var2, var3])

    assert d[var1] == 3.5
    assert d[var2] == 5
    assert d[var3] == 2.5


def test_linear_regression():
    xs = [(1, 1.5), (1.5, -1)]
    ys = [0, 1]
    
    def loss(w):
        wterms = [ad.Variable(wi) for wi in w]
        err_terms = []
        for x, cx in zip(xs, ys):
            wtx = sum((wt * ad.Constant(x) for wt, x in zip(wterms[1:], x)), wterms[0])
            diff = ad.Constant(cx) + ad.Negation(wtx)
            err_terms.append(diff * diff)
        return sum(err_terms[1:], err_terms[0]), wterms
    
    lterm, [w0, w1, w2] = loss((-1, 1.5, 0.5))

    assert lterm.compute() == 1.625
    d = ad.grad(lterm, [w0, w1, w2])
    print(lterm)
    print(d)
    assert d[w0] == 2
    assert d[w1] == 1.75
    assert d[w2] == 4.25


def test_logistic_regression():
    xs = [(1, 1.5), (1.5, -1)]
    ys = [0, 1]
    
    def loss(w):
        wterms = [ad.Variable(wi) for wi in w]
        err_terms = []
        for x, cx in zip(xs, ys):
            wtx = ad.Sigmoid(sum((wt * ad.Constant(x) for wt, x in zip(wterms[1:], x)), wterms[0]))
            diff = ad.Constant(cx) + ad.Constant(-1) * wtx
            err_terms.append(diff * diff)
        return sum(err_terms[1:], err_terms[0]), wterms
    
    lterm, [w0, w1, w2] = loss((-1, 1.5, 0.5))

    assert lterm.compute() == pytest.approx(0.71, abs=0.01)

    d = ad.grad(lterm, [w0, w1, w2])

    assert d[w0] == pytest.approx(.13, abs=.01)
    assert d[w1] == pytest.approx(.06, abs=.01)
    assert d[w2] == pytest.approx(.54, abs=.01)


def test_bgd_models_api():
    data = tr.as_training_set(
        [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]],
        [[1], [2], [3], [4], [5]]
    )

    model = md.Perceptron(in_dim=2, out_dim=1)
    model.initialize()

    loss = tr.RSSLoss()

    tr.batch_gradient_descent(model, loss, data, 1000, 0.01, None, data)

    ws = model.parameters()
    assert ws[0].value == pytest.approx(1, abs=0.001)
    assert ws[1].value + ws[2].value == pytest.approx(1, abs=0.001)

def test_bgd_xor_mlp():
    data = tr.as_training_set(
        [[0,0], [0, 1], [1, 0], [1, 1]],
        [[0], [1], [1], [0]]
    )

    model = md.MultiLayerPerceptron(2, [3], 1, [ad.ReLU, ad.Sigmoid])
    model.initialize()

    loss = tr.RSSLoss()

    tr.batch_gradient_descent(model, loss, data, 1000, 0.01)

    y = [model(x)[0].compute() for (x, _) in data]

    assert y[0] < 0.5
    assert y[1] > 0.5
    assert y[2] > 0.5
    assert y[3] < 0.5