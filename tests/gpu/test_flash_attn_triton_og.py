from math import sqrt

from torch.testing import assert_allclose

from src.functional import slow_attention
from src.flash_attn_triton_og import attention
from tests.common import get_query_key_value, device_name


def test_attention(device_name):
    batch_size = 32
    max_sequence_len = 1024
    embed_dimension = 32

    query, key, value = get_query_key_value(batch_size, max_sequence_len, embed_dimension, device=device_name)
    actual = attention(query, key, value, 1 / sqrt(query.size(-1)))
    expected = slow_attention(query, key, value)
    assert_allclose(actual, expected)
