from torch import float16
from torch.testing import assert_close

from src.functional import slow_attention
from src.flash_attn_triton_softmax1 import attention
from tests.common import get_query_key_value, device_name


def test_attention(device_name):
    batch_size = (32, 32)
    max_sequence_len = 1024
    embed_dimension = 32

    dtype=float16
    atol = 1e-3
    rtol = 0.

    query, key, value = get_query_key_value(batch_size, max_sequence_len, embed_dimension, device=device_name, dtype=dtype)
    actual = attention(query, key, value)
    expected = slow_attention(query, key, value, use_softmax1=True)
    assert_close(actual, expected, atol=atol, rtol=rtol)
