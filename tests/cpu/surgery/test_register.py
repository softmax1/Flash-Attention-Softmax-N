from types import MethodType

from pytest import mark, raises
from torch import Tensor, ones
from torch.nn import Module
from torch.testing import assert_close

from flash_attention_softmax_n import slow_attention_n, flash_attention_n
from flash_attention_softmax_n.surgery import apply_attention_softmax_n
from flash_attention_softmax_n.surgery.surgery_functions import policy_registry
from tests.common import device_name, attention_analytic_answer


SCALE = 0.2
FACTOR = 2.


class DummyModel(Module):
    def __init__(self):
        super().__init__()
        self.attn = DoubleAttention()

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return self.attn(query, key, value)


class DoubleAttention(Module):
    """
    Return 2x attention
    """
    def __init__(self):
        self.factor = FACTOR  # FACTOR == 2
        super().__init__()

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return self.factor * slow_attention_n(query, key, value, softmax_n_param=0, scale=SCALE)


@policy_registry.register(DoubleAttention)
def double_attention_converter(module: Module, module_index: int, softmax_n_param: float) -> Module:
    """
    Example: replace slow attention with flash attention.
    """
    assert isinstance(module, DoubleAttention)
    del module_index  # unused
    module.n = softmax_n_param
    setattr(module, 'forward', MethodType(forward, module))
    return module


def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    """
    New forward method.
    """
    return self.factor * flash_attention_n(query, key, value, softmax_n_param=int(self.n), scale=SCALE)


@mark.parametrize("weight", [10, 1, 0.1, -0.1, -1, -10])
def test_register(device_name, weight):
    N = 2
    L = 3
    S = 4
    E = 8
    Ev = 7

    query = weight * ones((N, L, E), device=device_name)
    key = weight * ones((N, S, E), device=device_name)
    value = weight * ones((N, S, Ev), device=device_name)

    slow_model = DummyModel()
    with raises(AttributeError):
        assert slow_model.n == 0
    actual_slow = slow_model(query, key, value)
    expected_0 = attention_analytic_answer(N, L, S, E, Ev, SCALE, weight, softmax_n_param=0, device=device_name, dtype=query.dtype)
    assert_close(actual_slow, FACTOR * expected_0)

    query = weight * ones((N, 1, L, E), device=device_name)
    key = weight * ones((N, 1, S, E), device=device_name)
    value = weight * ones((N, 1, S, Ev), device=device_name)

    flash_model = DummyModel()
    apply_attention_softmax_n(model=flash_model, softmax_n_param=0)
    assert flash_model.attn.n == 0
    actual_flash = flash_model(query, key, value)
    assert_close(actual_flash[:, 0, :, :], FACTOR * expected_0)

    flash_model_1 = DummyModel()
    apply_attention_softmax_n(model=flash_model_1, softmax_n_param=1)
    assert flash_model_1.attn.n == 1
    actual_flash_1 = flash_model_1(query, key, value)
    expected_1 = attention_analytic_answer(N, L, S, E, Ev, SCALE, weight, softmax_n_param=1, device=device_name, dtype=query.dtype)
    assert_close(actual_flash_1[:, 0, :, :], FACTOR * expected_1)
