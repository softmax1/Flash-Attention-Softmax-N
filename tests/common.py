from typing import Optional, Tuple, Union, Iterable

from pytest import fixture
from torch import Tensor, empty
from torch.cuda import is_available

from src.functional import DType


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
