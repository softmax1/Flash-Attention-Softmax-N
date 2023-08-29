from pytest import approx
from torch import Tensor, rand, randn, empty

from flash_attention_softmax_n.analysis.statistics import (
    central_moment,
    standardized_moment,
    variance,
    standard_deviation,
    skewness,
    kurtosis,
    variance_batch_mean,
    skewness_batch_mean,
    kurtosis_batch_mean
)


def test_central_moment():
    x = Tensor([5., 1., 0.])
    assert central_moment(x, 1).item() == approx(0., abs=1e-5)
    assert central_moment(x, 2).item() == approx((9 + 1 + 4) / 3, abs=1e-5)
    assert central_moment(x, 3).item() == approx((27 - 1 - 8) / 3, abs=1e-5)
    assert central_moment(x, 4).item() == approx((81 + 1 + 16) / 3, abs=1e-5)


def test_standardized_moment():
    xn = randn((1000, 1000))
    assert standardized_moment(xn, 1).item() == approx(0., abs=1e-7)
    assert standardized_moment(xn, 2).item() == approx(1., abs=1e-7)
    assert standardized_moment(xn, 3).item() == approx(0., abs=1e-2)
    assert standardized_moment(xn, 4).item() == approx(3., abs=1e-2)

    x = rand((1000, 1000))
    assert standardized_moment(x, 1).item() == approx(0., abs=1e-7)
    assert standardized_moment(x, 2).item() == approx(1., abs=1e-7)
    assert standardized_moment(x, 3).item() == approx(0., abs=1e-2)
    assert standardized_moment(x, 4).item() == approx(1.8, abs=1e-2)


def test_variance():
    xn = randn((1000, 1000))
    assert variance(xn).item() == approx(1., abs=1e-2)
    x = rand((1000, 1000))
    assert variance(x).item() == approx(1 / 12, abs=1e-2)


def test_standard_deviation():
    xn = randn((1000, 1000))
    assert standard_deviation(xn).item() == approx(1., abs=1e-2)
    x = rand((1000, 1000))
    assert standard_deviation(x).item() == approx(0.5 / 3**0.5, abs=1e-2)


def test_skewness():
    xln = empty((1000, 1000)).log_normal_()
    assert skewness(xln).item() > 0

    xn = randn((1000, 1000))
    assert skewness(xn).item() == approx(0., abs=1e-2)
    x = rand((1000, 1000))
    assert skewness(x).item() == approx(0., abs=1e-2)


def test_kurtosis():
    xn = randn((1000, 1000))
    assert kurtosis(xn).item() == approx(0., abs=1e-2)
    x = rand((1000, 1000))
    assert kurtosis(x).item() == approx(-1.2, abs=1e-2)


def test_variance_batch_mean():
    x = Tensor([
        [1., 2., 3., 4., 5.],
        [6., 7., 8., 9., 0.]
    ])
    assert variance(x[0]) == approx(2., abs=1e-2)
    assert variance(x[1]) == approx(10., abs=1e-2)
    assert variance_batch_mean(x) == approx(6., abs=1e-2)
    assert variance(x).item() == approx(2 * (81 + 49 + 25 + 9 + 1) / (4 * 10), abs=1e-2)
    assert variance_batch_mean(x) < variance(x).item()

    xln = empty((1000, 1000)).log_normal_()
    assert variance_batch_mean(xln) < variance(xln).item()
    xu = rand((1000, 1000))
    assert variance_batch_mean(xu) == approx(variance(xu).item(), abs=1e-2)
    xn = randn((1000, 1000))
    assert variance_batch_mean(xn) == approx(variance(xn).item(), abs=1e-2)


def test_skewness_batch_mean():
    x = Tensor([
        [1., 2., 3., 4., 5.],
        [6., 7., 8., 9., 0.]
    ])
    assert skewness(x[0]) == approx(0., abs=1e-2)
    assert skewness(x[1]) < 0.
    assert skewness_batch_mean(x) < 0.
    assert skewness(x) == approx(0., abs=1e-2)
    assert skewness_batch_mean(x) < skewness(x).item()

    xln = empty((1000, 1000)).log_normal_()
    assert skewness_batch_mean(xln) < skewness(xln).item()
    xu = rand((1000, 1000))
    assert skewness_batch_mean(xu) == approx(skewness(xu).item(), abs=1e-2)
    xn = randn((1000, 1000))
    assert skewness_batch_mean(xn) == approx(skewness(xn).item(), abs=1e-2)


def test_kurtosis_batch_mean():
    x = Tensor([
        [1., 2., 3., 4., 5.],
        [6., 7., 8., 9., 0.]
    ])
    assert kurtosis(x[0]).item() == approx(-1.2, abs=0.1)
    assert kurtosis(x[1]) > kurtosis(x[0])
    assert kurtosis_batch_mean(x) > -1.2
    assert kurtosis(x).item() == approx(-1.2, abs=0.1)
    assert kurtosis_batch_mean(x) > kurtosis(x).item()

    xln = empty((1000, 1000)).log_normal_()
    assert 10 * kurtosis_batch_mean(xln) < kurtosis(xln).item()  # big differences are possible
    xu = rand((1000, 1000))
    assert kurtosis_batch_mean(xu) == approx(kurtosis(xu).item(), abs=1e-2)
    xn = randn((1000, 1000))
    assert kurtosis_batch_mean(xn) == approx(kurtosis(xn).item(), abs=1e-2)
