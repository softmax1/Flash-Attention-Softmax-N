from pytest import approx, raises
from torch import Tensor, log, float16
from torch.cuda import is_available
from torch.nn.functional import scaled_dot_product_attention
from torch.testing import assert_allclose

from src.functional import softmax_1, slow_attention
from tests.common import get_query_key_value


def test_softmax_1():
    expected_numerators = [
        [1, 3, 6],
        [3, 1, 4],
        [1 / 6, 1 / 3, 1 / 2],
        [0.5, 1.5, 3],
        [100, 200, 300],
        [1 / 600, 1 / 300, 1 / 200],
        [2 / 7, 4 / 7, 8 / 7]
    ]
    expected_denominators = [sum(test_case) for test_case in expected_numerators]
    input_data = log(Tensor(expected_numerators))

    output_1 = softmax_1(input_data, dim=-1)
    assert output_1.size() == input_data.size()
    for idx in range(input_data.size(0)):
        for jdx in range(input_data.size(1)):
            expected_answer = expected_numerators[idx][jdx] / (expected_denominators[idx] + 1)
            assert output_1[idx][jdx].item() == approx(expected_answer)

    # exp([12., 89., 710.]) will lead to overflow at half-, single-, or double-precision
    overflow_test_input = Tensor([12., 89., 710.])
    finite_ouput = softmax_1(overflow_test_input, dim=-1)
    assert finite_ouput.sum() == 1


def test_slow_attention():
    batch_size = 2
    max_sequence_len = 3
    embed_dimension = 8
    device = "cuda" if is_available() else "cpu"

    with raises(RuntimeError):
        query, key, value = get_query_key_value(batch_size, max_sequence_len, embed_dimension, device=device, dtype=float16)
        scaled_dot_product_attention(query, key, value)

    query, key, value = get_query_key_value(batch_size, max_sequence_len, embed_dimension, device=device)
    actual = slow_attention(query, key, value)
    expected = scaled_dot_product_attention(query, key, value)
    assert_allclose(actual, expected)
