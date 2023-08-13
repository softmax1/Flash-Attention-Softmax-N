import gc

from pytest import mark
from torch import float16, randn_like, ones
from torch.cuda import empty_cache
from torch.testing import assert_close

from src.functional import slow_attention_n
from src.flash_attn_triton import flash_attention_n
from tests.common import get_query_key_value, device_name, attention_analytic_answer, attention_analytic_casual_answer


@mark.parametrize("sm_n", [0., 1., 1e-3, 1e-6, 4.])
@mark.parametrize("is_causal", [False, True])
@mark.parametrize("scale", [None, 0.5, 0.01])
def test_flash_attention_comparison(device_name, sm_n, is_causal, scale):
    """
    Compare the Torch and Triton Attention implementations
    """
    batch_size = (32, 32)
    max_sequence_len = 1024
    embed_dimension = 32

    atol = 1e-2 if is_causal else 1e-3
    rtol = 0.

    # Test forward step,
    query, key, value = get_query_key_value(batch_size, max_sequence_len, embed_dimension, device=device_name, dtype=float16)
    actual = flash_attention_n(query, key, value, causal=is_causal, sm_scale=scale, sm_n=sm_n)
    expected = slow_attention_n(query, key, value, is_causal=is_causal, scale=scale, n=sm_n)
    assert_close(actual, expected, atol=atol, rtol=rtol)

    # and backward step.
    doutput = randn_like(actual)
    actual.backward(doutput)
    actual_dvalue, value.grad = value.grad.clone(), None
    actual_dkey, key.grad = key.grad.clone(), None
    actual_dquery, query.grad = query.grad.clone(), None
    expected.backward(doutput)
    expected_dvalue, value.grad = value.grad.clone(), None
    expected_dkey, key.grad = key.grad.clone(), None
    expected_dquery, query.grad = query.grad.clone(), None
    assert_close(actual_dvalue, expected_dvalue, atol=atol, rtol=rtol)
    assert_close(actual_dkey, expected_dkey, atol=atol, rtol=rtol)
    assert_close(actual_dquery, expected_dquery, atol=atol, rtol=rtol)

    empty_cache()
    gc.collect()


@mark.parametrize("sm_n", [0., 1., 1e-3, 1e-6, 4.])
@mark.parametrize("weight", [10, 3, 0.5, 0.04, 0.02, 0.01, 0, -0.01, -0.02, -0.04, -0.5, -3, -10])
def test_flash_attention_analytic(device_name, sm_n, weight):
    """
    When the elements of the input tensors Q, K, & V all have the same value, there is a closed-form expression for the output of Attention w/ or w/o a casual mask.
    """
    N = 6
    L = 1024
    S = 1024 + 128
    E = 64
    Ev = 64
    scale = 0.3

    atol = 1e-3
    rtol = 0.

    query = weight * ones((N, 1, L, E), device=device_name, dtype=float16)
    key = weight * ones((N, 1, S, E), device=device_name, dtype=float16)
    value = weight * ones((N, 1, S, Ev), device=device_name, dtype=float16)

    output_0a = slow_attention_n(query, key, value, scale=scale, n=sm_n)
    output_1a = flash_attention_n(query, key, value, sm_scale=scale, sm_n=sm_n)

    expected_a = attention_analytic_answer(N, L, S, E, Ev, scale, weight, softmax_n_param=sm_n, device=device_name)

    assert_close(output_1a, output_0a, atol=atol, rtol=rtol)
    assert_close(output_0a, expected_a, atol=atol, rtol=rtol)
    assert_close(output_1a, expected_a, atol=atol, rtol=rtol)

    atol = 0
    rtol = 2e-3

    output_0b = slow_attention_n(query, key, value, scale=scale, is_causal=True, n=sm_n)
    output_1b = flash_attention_n(query, key, value, causal=True, sm_scale=scale, sm_n=sm_n)

    expected_b = attention_analytic_casual_answer(N, L, S, E, Ev, scale, weight, softmax_n_param=sm_n, device=device_name)

    assert_close(output_1b, output_0b, atol=atol, rtol=rtol)
    assert_close(output_0b.sum(dim=0).sum(dim=-1), expected_b, atol=atol, rtol=rtol)
    assert_close(output_1b.sum(dim=0).sum(dim=-1), expected_b, atol=atol, rtol=rtol)

    empty_cache()
    gc.collect()
