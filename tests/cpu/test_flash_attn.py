from pytest import mark
from torch import randn_like, ones
from torch.testing import assert_close

from src.flash_attn import flash_attention_n
from src.functional import slow_attention_n
from tests.common import get_query_key_value, device_name, attention_analytic_answer, attention_analytic_casual_answer


@mark.parametrize("n", [0, 1, 4])
@mark.parametrize("scale", [None, 0.1, 0.5])
@mark.parametrize("dropout", [0., 0.2])
@mark.parametrize("is_causal", [False, True])
def test_flash_attention_n(device_name, n, scale, dropout, is_causal):
    batch_size = (2, 1)
    max_sequence_len = 3
    embed_dimension = 8

    atol = 1e-6
    rtol = 0.

    # Test forward step,
    query, key, value = get_query_key_value(batch_size, max_sequence_len, embed_dimension, device=device_name)
    expected = slow_attention_n(query, key, value, n=n, scale=scale, dropout_p=dropout, is_causal=is_causal)
    actual = flash_attention_n(query, key, value, softmax_n_param=n, scale=scale, dropout_p=dropout, is_causal=is_causal)

    if is_causal and n > 0:
        print(n, scale, dropout, is_causal)
        print(expected)
        print(actual)

    if dropout > 0.:
        assert isinstance(actual.sum().item(), float) != 0 and isinstance(expected.sum().item(), float)
    else:
        assert_close(actual, expected, atol=atol, rtol=rtol)

    # and backward step.
    doutput = randn_like(query)
    expected.backward(doutput)
    expected_dvalue, value.grad = value.grad.clone(), None
    expected_dkey, key.grad = key.grad.clone(), None
    expected_dquery, query.grad = query.grad.clone(), None
    actual.backward(doutput)
    actual_dvalue, value.grad = value.grad.clone(), None
    actual_dkey, key.grad = key.grad.clone(), None
    actual_dquery, query.grad = query.grad.clone(), None
    if dropout > 0.:
        assert isinstance(actual_dvalue.sum().item(), float) != 0 and isinstance(expected_dvalue.sum().item(), float)
        assert isinstance(actual_dkey.sum().item(), float) != 0 and isinstance(expected_dkey.sum().item(), float)
        assert isinstance(actual_dquery.sum().item(), float) != 0 and isinstance(expected_dquery.sum().item(), float)
    else:
        assert_close(actual_dvalue, expected_dvalue, atol=atol, rtol=rtol)
        assert_close(actual_dkey, expected_dkey, atol=atol, rtol=rtol)
        assert_close(actual_dquery, expected_dquery, atol=atol, rtol=rtol)


@mark.parametrize("n", [0, 1, 4])
@mark.parametrize("weight", [10, 3, 0.5, 0.04, 0.02, 0.01, 0, -0.01, -0.02, -0.04, -0.5, -3, -10])
def test_flash_attention_analytic(device_name, n, weight):
    """
    When the elements of the input tensors Q, K, & V all have the same value, there is a closed-form expression for the output of Attention w/ or w/o a casual mask.
    """
    N = 2
    L = 2
    S = 2
    E = 8
    Ev = 8
    scale = 0.3

    atol = 1e-3
    rtol = 0.

    query = weight * ones((N, 1, L, E), device=device_name)
    key = weight * ones((N, 1, S, E), device=device_name)
    value = weight * ones((N, 1, S, Ev), device=device_name)

    output_0a = slow_attention_n(query, key, value, scale=scale, n=n)
    output_1a = flash_attention_n(query, key, value, scale=scale, softmax_n_param=n)

    expected_a = attention_analytic_answer(N, L, S, E, Ev, scale, weight, softmax_n_param=n, device=device_name, dtype=query.dtype)

    assert_close(output_1a, output_0a, atol=atol, rtol=rtol)
    assert_close(output_0a[:, 0, :, :], expected_a, atol=atol, rtol=rtol)
    assert_close(output_1a[:, 0, :, :], expected_a, atol=atol, rtol=rtol)

    atol = 0
    rtol = 2e-3

    output_0b = slow_attention_n(query, key, value, scale=scale, is_causal=True, n=n)
    output_1b = flash_attention_n(query, key, value, scale=scale, is_causal=True, softmax_n_param=n)

    expected_b = attention_analytic_casual_answer(N, L, S, E, Ev, scale, weight, softmax_n_param=n, device=device_name, dtype=query.dtype)

    assert_close(output_1b, output_0b, atol=atol, rtol=rtol)
    assert_close(output_0b.sum(dim=0).sum(dim=-1)[0], expected_b, atol=atol, rtol=rtol)
    assert_close(output_1b.sum(dim=0).sum(dim=-1)[0], expected_b, atol=atol, rtol=rtol)
