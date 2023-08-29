from typing import Tuple, Optional, Union

from torch import Tensor, mean, pow, no_grad


_Dim = Optional[Union[int, Tuple[int]]]


@no_grad()
def central_moment(x: Tensor, k: int, dim: _Dim = None) -> Tensor:
    """
    Moment of a random variable about the random variable's mean.
    """
    return mean(pow(x - mean(x, dim=dim, keepdim=True), k), dim=dim)


@no_grad()
def variance(x: Tensor, dim: _Dim = None) -> Tensor:
    return central_moment(x, 2, dim=dim)


@no_grad()
def standard_deviation(x: Tensor, dim: _Dim = None) -> Tensor:
    return pow(variance(x, dim=dim), 0.5)


@no_grad()
def standardized_moment(x: Tensor, k: int, dim: _Dim = None) -> Tensor:
    """
    The ratio of the kth moment about the mean.
    """
    return central_moment(x, k, dim=dim) / pow(variance(x, dim=dim), k / 2)


@no_grad()
def skewness(x: Tensor, dim: _Dim = None) -> Tensor:
    return standardized_moment(x, 3, dim=dim)


@no_grad()
def kurtosis(x: Tensor, dim: _Dim = None) -> Tensor:
    """
    Excess kurtosis
    """
    return standardized_moment(x, 4, dim=dim) - 3.


def _get_stat_dim(x: Tensor) -> Tuple[int]:
    """
    If no `dim` is specified, assume `x` is a batch of samples and compute the statistic for each sample.
    """
    return tuple(range(1, x.ndim))


@no_grad()
def variance_batch_mean(x: Tensor) -> float:
    """
    Compute the variance for each sample in a batch, and then compute the mean of the variance in the batch.
    """
    var = variance(x, dim=_get_stat_dim(x))
    return mean(var).item()


@no_grad()
def skewness_batch_mean(x: Tensor) -> float:
    """
    Compute the skewness for each sample in a batch, and then compute the mean of the skewness in the batch.
    """
    skew = skewness(x, dim=_get_stat_dim(x))
    return mean(skew).item()


@no_grad()
def kurtosis_batch_mean(x: Tensor) -> float:
    """
    Compute the kurtosis for each sample in a batch, and then compute the mean of the kurtosis in the batch.
    """
    kurt = kurtosis(x, dim=_get_stat_dim(x))
    return mean(kurt).item()
