from pytest import approx, raises, mark
from torch import Tensor, log, float16, randn_like, ones
from torch.nn.functional import scaled_dot_product_attention
from torch.testing import assert_close

from src.functional import softmax_n, slow_attention_n
from tests.common import get_query_key_value, device_name, attention_analytic_answer, attention_analytic_casual_answer


@mark.parametrize("n", [0., 1., 1e-3, 1e-6, 4.])
def test_softmax_n(device_name, n):
    """
    Tests based on https://github.com/softmax1/EsperBERTo/blob/main/tests/functional/test_core.py
    """
    expected_numerators = [
        [1, 3, 6],
        [3, 1, 4],
        [1 / 6, 1 / 3, 1 / 2],
        [0.5, 1.5, 3],
        [100, 200, 300],
        [1 / 600, 1 / 300, 1 / 200],
        [2 / 7, 4 / 7, 8 / 7]
    ]
    input_1 = log(Tensor(expected_numerators).to(device_name))
    expected_denominators = [n + sum(test_case) for test_case in expected_numerators]

    output_1 = softmax_n(input_1, n=n, dim=-1)
    assert output_1.size() == input_1.size()
    for idx in range(input_1.size(0)):
        for jdx in range(input_1.size(1)):
            expected_answer = expected_numerators[idx][jdx] / expected_denominators[idx]
            assert output_1[idx][jdx].item() == approx(expected_answer)

    # exp([12., 89., 710.]) will naively lead to overflow at half-, single-, or double-precision
    input_2 = Tensor([12., 89., 710.]).to(device_name)
    output_2 = softmax_n(input_2, n, dim=-1)
    assert output_2.sum() == 1


def test_slow_attention_0(device_name):
    """
    Comparing my Attention implementation to torch's `scaled_dot_product_attention`.
    It's less than ideal though because that is an experimental function.
    The assumption is if my attention implementation is correct for softmax_0, and my softmax_n implementation is correct (see `test_softmax_n`), then my implementation of attention with softmax_n will also be correct.
    """
    batch_size = 2
    max_sequence_len = 3
    embed_dimension = 8

    atol = 1e-6
    rtol = 0.

    query_0, key_0, value_0 = get_query_key_value(batch_size, max_sequence_len, embed_dimension, device=device_name, dtype=float16)
    if "cuda" in device_name:
        actual_0 = slow_attention_n(query_0, key_0, value_0)
        expected_0 = scaled_dot_product_attention(query_0, key_0, value_0)
        atol_f16 = atol**0.5
        assert_close(actual_0, expected_0, atol=atol_f16, rtol=rtol)
    else:
        with raises(RuntimeError):
            slow_attention_n(query_0, key_0, value_0)
        with raises(RuntimeError):
            scaled_dot_product_attention(query_0, key_0, value_0)

    # Test forward step,
    query_1, key_1, value_1 = get_query_key_value(batch_size, max_sequence_len, embed_dimension, device=device_name)
    actual_1 = slow_attention_n(query_1, key_1, value_1)
    expected_1 = scaled_dot_product_attention(query_1, key_1, value_1)
    assert_close(actual_1, expected_1, atol=atol, rtol=rtol)

    # and backward step.
    doutput_1 = randn_like(query_1)
    actual_1.backward(doutput_1)
    actual_dvalue_1, value_1.grad = value_1.grad.clone(), None
    actual_dkey_1, key_1.grad = key_1.grad.clone(), None
    actual_dquery_1, query_1.grad = query_1.grad.clone(), None
    expected_1.backward(doutput_1)
    expected_dvalue_1, value_1.grad = value_1.grad.clone(), None
    expected_dkey_1, key_1.grad = key_1.grad.clone(), None
    expected_dquery_1, query_1.grad = query_1.grad.clone(), None
    assert_close(actual_dvalue_1, expected_dvalue_1, atol=atol, rtol=rtol)
    assert_close(actual_dkey_1, expected_dkey_1, atol=atol, rtol=rtol)
    assert_close(actual_dquery_1, expected_dquery_1, atol=atol, rtol=rtol)

    # torch version doesn't have a scale argument
    with raises(TypeError):
        scaled_dot_product_attention(query_1, key_1, value_1, scale=0.1)

    # Trying to test dropout. There's probably a better way to do it.
    query_2, key_2, value_2 = get_query_key_value(batch_size, max_sequence_len, embed_dimension, device=device_name)
    dropout_p = 0.25
    actual_2 = slow_attention_n(query_2, key_2, value_2, dropout_p=dropout_p)
    expected_2 = scaled_dot_product_attention(query_2, key_2, value_2, dropout_p=dropout_p)
    assert actual_2.sum() != expected_2.sum()

    # Testing casual mask.
    query_3, key_3, value_3 = get_query_key_value(batch_size, max_sequence_len, embed_dimension, device=device_name)
    actual_3 = slow_attention_n(query_3, key_3, value_3, is_causal=True)
    expected_3 = scaled_dot_product_attention(query_3, key_3, value_3, is_causal=True)
    assert_close(actual_3, expected_3, atol=atol, rtol=rtol)

    # Test boolean attention mask.
    query_4, key_4, value_4 = get_query_key_value(batch_size, max_sequence_len, embed_dimension, device=device_name)
    attn_mask_1 = Tensor([
        [[True for _ in range(max_sequence_len)] for _ in range(max_sequence_len)],
        [[False for _ in range(max_sequence_len)] for _ in range(max_sequence_len)]
    ]).bool().to(device_name)
    actual_4 = slow_attention_n(query_4, key_4, value_4, attn_mask=attn_mask_1)
    expected_4 = scaled_dot_product_attention(query_4, key_4, value_4, attn_mask=attn_mask_1)
    assert_close(actual_4, expected_4, atol=atol, rtol=rtol)

    # Testing float attention mask.
    query_5, key_5, value_5 = get_query_key_value(batch_size, max_sequence_len, embed_dimension, device=device_name)
    attn_mask_2 = Tensor([[0.1, 0.2, 0.3] for _ in range(max_sequence_len)]).to(device_name)
    actual_5 = slow_attention_n(query_5, key_5, value_5, attn_mask=attn_mask_2)
    expected_5 = scaled_dot_product_attention(query_5, key_5, value_5, attn_mask=attn_mask_2)
    assert_close(actual_5, expected_5, atol=atol, rtol=rtol)


@mark.parametrize("sm_n", [0., 1., 1e-3, 1e-6, 4.])
@mark.parametrize("weight", [10, 1, 0.1, -0.1, -1, 10])
def test_slow_attention_n(device_name, sm_n, weight):
    """
    When the elements of the input tensors Q, K, & V all have the same value, there is a closed-form expression for the output of Attention w/ or w/o a casual mask.
    """
    N = 2
    L = 3
    S = 4
    E = 8
    Ev = 7
    scale = 0.3

    query = weight * ones((N, L, E), device=device_name)
    key = weight * ones((N, S, E), device=device_name)
    value = weight * ones((N, S, Ev), device=device_name)

    output_a = slow_attention_n(query, key, value, scale=scale, n=sm_n)
    expected_a = attention_analytic_answer(N, L, S, E, Ev, scale, weight, softmax_n_param=sm_n, device=device_name)
    assert_close(output_a, expected_a)

    output_b = slow_attention_n(query, key, value, scale=scale, is_causal=True, n=sm_n)
    expected_b = attention_analytic_casual_answer(N, L, S, E, Ev, scale, weight, softmax_n_param=sm_n, device=device_name)
    assert_close(output_b.sum(dim=0).sum(dim=-1), expected_b)
