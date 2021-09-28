import pytest
from diy_autodiff import __version__
from diy_autodiff import autodiff as ad


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
    
    def L(w):
        wterms = [ad.Variable(wi) for wi in w]
        err_terms = []
        for x, cx in zip(xs, ys):
            wtx = sum((wt * ad.Constant(x) for wt, x in zip(wterms[1:], x)), wterms[0])
            diff = ad.Constant(cx) + ad.Negation(wtx)
            err_terms.append(diff * diff)
        return sum(err_terms[1:], err_terms[0]), wterms
    
    lterm, [w0, w1, w2] = L((-1, 1.5, 0.5))

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
    
    def L(w):
        wterms = [ad.Variable(wi) for wi in w]
        err_terms = []
        for x, cx in zip(xs, ys):
            wtx = ad.Sigmoid(sum((wt * ad.Constant(x) for wt, x in zip(wterms[1:], x)), wterms[0]))
            diff = ad.Constant(cx) + ad.Constant(-1) * wtx
            err_terms.append(diff * diff)
        return sum(err_terms[1:], err_terms[0]), wterms
    
    lterm, [w0, w1, w2] = L((-1, 1.5, 0.5))

    assert lterm.compute() == pytest.approx(0.71, abs=0.01)

    d = ad.grad(lterm, [w0, w1, w2])

    assert d[w0] == pytest.approx(.13, abs=.01)
    assert d[w1] == pytest.approx(.06, abs=.01)
    assert d[w2] == pytest.approx(.54, abs=.01)


def test_igd():
    xs = [(0,0), (1,1), (2,2), (3,3), (4,4)]
    cs = [1, 2, 3, 4, 5]
    ws = [-1, 2 ,3]

    # change to constants / variables
    xs = [tuple(ad.Constant(xi) for xi in x) for x in xs]
    cs = [ad.Constant(c) for c in cs]
    ws = [ad.Variable(w) for w in ws]

    # define linear regression model with rss loss
    loss = ad.Constant(0)
    for x, cx in zip(xs, cs):
        y = sum((wi*xi for wi, xi in zip(ws, (ad.Constant(1),) + x)), ad.Constant(0))
        loss += ad.Square(cx - y)

    ad.gradient_descent(loss, ws, 0.01, 1000)

    assert ws[0].value == pytest.approx(1, abs=0.001)
    assert ws[1].value + ws[2].value == pytest.approx(1, abs=0.001)