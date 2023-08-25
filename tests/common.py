from math import exp
from typing import Optional, Tuple, Union, Iterable

from pytest import fixture
from torch import Tensor, empty, ones
from torch.cuda import is_available

from flash_attention_softmax_n.functional import DType


def get_query_key_value(batch_size: Union[int, Iterable[int]],
                        max_sequence_len: int,
                        embed_dimension: int,
                        device: Optional[str] = None,
                        dtype: Optional[DType] = None
                        ) -> Tuple[Tensor, Tensor, Tensor]:
    shape = (batch_size, max_sequence_len, embed_dimension) if isinstance(batch_size, int) else (*batch_size, max_sequence_len, embed_dimension)
    query = empty(shape, device=device, dtype=dtype).normal_(mean=0., std=0.5).requires_grad_()
    key = empty(shape, device=device, dtype=dtype).normal_(mean=0., std=0.5).requires_grad_()
    value = empty(shape, device=device, dtype=dtype).normal_(mean=0., std=0.5).requires_grad_()
    return query, key, value


@fixture(scope='session')
def device_name() -> str:
    return "cuda" if is_available() else "cpu"


def attention_analytic_answer(N: int, L: int, S: int, E: int, Ev: int,
                              scale: float, weight: float, softmax_n_param: float,
                              device: str, dtype: DType
                              ) -> Tensor:
    answer_0 = weight * ones((N, L, Ev), device=device)
    factor_n = S / (softmax_n_param * exp(-weight**2 * E * scale) + S)
    return (answer_0 * factor_n).type(dtype=dtype)


def attention_analytic_casual_answer(N: int, L: int, S: int, E: int, Ev: int,
                                     scale: float, weight: float, softmax_n_param: float,
                                     device: str, dtype: DType
                                     ) -> Tensor:
    factors_n = [(ell + S - L) / (softmax_n_param * exp(-weight**2 * E * scale) + (ell + S - L)) for ell in range(1, L + 1)]
    answer = N * Ev * weight * Tensor(factors_n).to(device=device)
    return answer.type(dtype=dtype)
