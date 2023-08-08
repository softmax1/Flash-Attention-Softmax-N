from math import sqrt
from typing import Optional

from torch import Tensor, index_select, arange, zeros, ones, bool, dropout
from torch.cuda import is_available
from torch.nn.functional import softmax, pad

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch.types import _dtype as DType
else:
    # The JIT doesn't understand Union, nor torch.dtype here
    DType = int


def softmax_1(input: Tensor, dim: Optional[int] = None, dtype: Optional[DType] = None) -> Tensor:
    """
    $\text(softmax)_n(x_i) = exp(x_i) / (1 + sum_j exp(x_j))$

    The idea here is to pad the input with zeros.
    That way the softmax, with its stable implementation under the hood, can naturally be used.
    Afterwards we need to un-pad the output.
    """
    if dim is None:
        raise NotImplementedError('The padding approach is currently only implemented for a specific dimension.')
    if dim >= 0:
        dim -= len(input.size())
    padding_size = -(2 * dim + 1) * (0,) + (1,)  # change the rightmost '1' to 'n' for softmax_n

    padded_input = pad(input, padding_size, value=0)
    padded_output = softmax(padded_input, dim=dim)
    device = 'cuda' if is_available() else 'cpu'
    indices_to_keep = arange(input.size(dim), device=device)
    output = index_select(padded_output, dim=dim, index=indices_to_keep)
    return output if dtype is None else output.type(dtype=dtype)


def slow_attention(query: Tensor,
                   key: Tensor,
                   value: Tensor,
                   attn_mask: Optional[Tensor] = None,
                   dropout_p: float = 0.0,
                   is_causal: bool = False,
                   scale: Optional[float] = None,
                   use_softmax1: bool = False
                   ) -> Tensor:
    """
    Inefficient implementation of Scaled Dot Product Attention

    Computes scaled dot product attention on query, key and value tensors, using an optional attention mask if passed, and applying dropout if a probability greater than 0.0 is specified.
    For more info see: https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html

    :param query: Query tensor; shape (N, ..., L, E).
    :param key: Key tensor; shape (N, ..., S, E).
    :param value: Value tensor; shape (N, ..., S, Ev).
    :param attn_mask: Attention mask; shape (N, ..., L, S). Two types of masks are supported. A boolean mask where a value of True indicates that the element should take part in attention. A float mask of the same type as query, key, value that is added to the attention score.
    :param dropout_p: Dropout probability; if greater than 0.0, dropout is applied
    :param is_causal: If true, assumes causal attention masking and errors if both attn_mask and is_causal are set.
    :param scale: Scaling factor applied prior to softmax. If None, the default value is set to 1 / sqrt(E).
    :param use_softmax1: If true, use softmax_1 instead of softmax_0
    :return: Attention output; shape (N, ..., L, Ev).

    Shape Legend:
    - N: batch size
    - S: source sequence length
    - L: target sequence length
    - E: embedding dimension of the query and key
    - Ev: embedding dimension of the value
    """
    device = 'cuda' if is_available() else 'cpu'
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / sqrt(query.size(-1)) if scale is None else scale
    attn_bias = zeros(L, S, dtype=query.dtype, device=device)
    if is_causal:
        assert attn_mask is None
        temp_mask = ones(L, S, dtype=bool, device=device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    if use_softmax1:
        attn_weight = softmax_1(attn_weight, dim=-1)
    else:
        attn_weight = softmax(attn_weight, dim=-1)
    attn_weight = dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value
