from typing import Optional, Tuple

from torch import Tensor, rand

from src.functional import DType


def get_query_key_value(batch_size: int,
                        max_sequence_len: int,
                        embed_dimension: int,
                        device: Optional[str] = None,
                        dtype: Optional[DType] = None
                        ) -> Tuple[Tensor, Tensor, Tensor]:
    query = rand(batch_size, max_sequence_len, embed_dimension, device=device, dtype=dtype)
    key = rand(batch_size, max_sequence_len, embed_dimension, device=device, dtype=dtype)
    value = rand(batch_size, max_sequence_len, embed_dimension, device=device, dtype=dtype)
    return query, key, value
