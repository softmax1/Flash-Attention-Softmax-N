from math import sqrt
from typing import Optional

from torch import Tensor, zeros, ones, dropout, exp
from torch import bool as torch_bool

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch.types import _dtype as DType
else:
    # The JIT doesn't understand Union, nor torch.dtype here
    DType = int


def softmax_n(x: Tensor, n: Optional[float] = None, dim: Optional[int] = None, dtype: Optional[DType] = None) -> Tensor:
    """
    $\text(softmax)_n(x_i) = exp(x_i) / (n + \sum_j exp(x_j))$

    Note: softmax_n, with fixed input, is _not_ shift-symmetric when n != 0, and we must account for this.
    Normally when computing a softmax, the maxes are subtracted from the inputs for numeric stability.
    """
    if n is None:
        n = 0.
    if dim is None:
        dim = -1
    shift = x.max(dim=dim, keepdim=True).values.detach()
    numerator = exp(x - shift)
    output = numerator / (n * exp(-shift) + numerator.sum(dim=dim, keepdim=True))
    return output if dtype is None else output.type(dtype=dtype)


def slow_attention_n(query: Tensor,
                     key: Tensor,
                     value: Tensor,
                     attn_mask: Optional[Tensor] = None,
                     dropout_p: float = 0.0,
                     is_causal: bool = False,
                     scale: Optional[float] = None,
                     softmax_n_param: Optional[float] = None,
                     softmax_dtype: Optional[DType] = None,
                     train: bool = True,
                     ) -> Tensor:
    """
    Inefficient implementation of Scaled Dot Product Attention

    Computes scaled dot product attention on query, key and value tensors, using an optional attention mask if passed, and applying dropout if a probability greater than 0.0 is specified.
    For more info see: https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html

    :param query: Query tensor; shape (N, ..., L, E).
    :param key: Key tensor; shape (N, ..., S, E).
    :param value: Value tensor; shape (N, ..., S, Ev).
    :param attn_mask: Attention mask; Two types of masks are supported.
        A boolean mask, shape (N, ..., L, S), where a value of True indicates that the element should take part in attention.
        A float mask, shape (L, S), of the same type as query, key, value that is added to the attention score.
    :param dropout_p: Dropout probability; if greater than 0.0, dropout is applied
    :param is_causal: If true, assumes causal attention masking and errors if both attn_mask and is_causal are set.
    :param scale: Scaling factor applied prior to softmax. If None, the default value is set to 1 / sqrt(E).
    :param softmax_n_param: Regularization parameter for the generalized softmax_n
    :param softmax_dtype: The datatype for the output from the softmax operation.
    param train: If false, turns off dropout for inference.
    :return: Attention output; shape (N, ..., L, Ev).

    Shape Legend:
    - N: batch size
    - S: source sequence length
    - L: target sequence length
    - E: embedding dimension of the query and key
    - Ev: embedding dimension of the value
    """
    if softmax_n_param is None:
        softmax_n_param = 0.
    if softmax_dtype is None:
        softmax_dtype = query.dtype

    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / sqrt(query.size(-1)) if scale is None else scale
    attn_bias = zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = ones(L, S, dtype=torch_bool, device=query.device).tril(diagonal=S-L)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch_bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = softmax_n(attn_weight, n=softmax_n_param, dim=-1, dtype=softmax_dtype)
    attn_weight = dropout(attn_weight, dropout_p, train=train)
    return attn_weight @ value
