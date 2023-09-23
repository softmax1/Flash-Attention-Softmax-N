from collections import namedtuple
from math import sqrt
from typing import Optional

from einops import rearrange
from torch import Tensor, ones, finfo, zeros
from torch import bool as torch_bool
from torch import device as device_obj
from torch.cuda import is_available, get_device_properties
from torch.backends.cuda import sdp_kernel
from torch.nn.functional import pad, scaled_dot_product_attention


_EfficientAttentionConfig = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])


def _flash_attn_config(query: Tensor) -> _EfficientAttentionConfig:
    """
    determine efficient attention configs for cuda and cpu
    """

    cpu_config = _EfficientAttentionConfig(True, True, True)
    cuda_config = None

    if is_available():
        device_properties = get_device_properties(device_obj('cuda'))

        if device_properties.major in {7, 8} and device_properties.minor == 0:
            # A100 (and A30) GPU == 8.0, V100 GPU = 7.0
            cuda_config = _EfficientAttentionConfig(True, False, False)
        else:
            # Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda
            cuda_config = _EfficientAttentionConfig(False, True, True)

    return cuda_config if query.is_cuda else cpu_config


def _create_causal_mask(i: int, j: int, device: device_obj) -> Tensor:
    return ones((i, j), device=device, dtype=torch_bool).triu(j - i + 1)


def flash_attention_n(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        softmax_n_param: Optional[int] = None,
        scale: Optional[float] = None,
        dropout_p: float = 0.,
        attn_mask: Optional[Tensor] = None,
        attn_bias: Optional[Tensor] = None,
        is_causal: bool = False
) -> Tensor:
    """
    CUDA implementation of Flash Attention with Softmax_n inspired by x-transformers
    :param query: Query tensor; shape (N, ..., L, E).
    :param key: Key tensor; shape (N, ..., S, E).
    :param value: Value tensor; shape (N, ..., S, Ev).
    :param softmax_n_param: Regularization parameter for the generalized softmax_n.
    :param scale: Scaling factor applied prior to softmax. If None, the default value is set to 1 / sqrt(E).
    :param dropout_p: Dropout probability; if greater than 0.0, dropout is applied
    :param attn_mask: Attention mask; shape (N, ..., L, S)
    :param attn_bias: ALiBi positional bias; shape(..., L, S)
    :param is_causal: If true, assumes causal attention masking.
    :return: Attention output; shape (N, ..., L, Ev).
    """
    if softmax_n_param is not None and softmax_n_param > 0:
        key, value = map(lambda t: pad(t, (0, 0, softmax_n_param, 0), value=0.), (key, value))

        if attn_mask is not None:
            attn_mask = pad(attn_mask, (softmax_n_param, 0), value=True)

        if attn_bias is not None:
            attn_bias = pad(attn_bias, (softmax_n_param, 0), value=0.)

    if key.ndim == 3:
        key = rearrange(key, 'b ... -> b 1 ...').expand_as(query)

    if value.ndim == 3:
        value = rearrange(value, 'b ... -> b 1 ...').expand_as(query)

    if scale is not None:
        default_scale = 1 / sqrt(query.shape[-1])
        query = query * (scale / default_scale)

    batch, heads, q_len, _, k_len, is_cuda, device, dtype = *query.shape, key.shape[-2], query.is_cuda, query.device, query.dtype

    if attn_mask is not None:
        assert attn_mask.ndim == 4
        attn_mask = attn_mask.expand(batch, heads, q_len, k_len)

        if is_causal:
            causal_mask = _create_causal_mask(q_len, k_len, device)
            attn_mask = attn_mask & ~causal_mask
            is_causal = False

    # the built-in argument `is_causal` of `scaled_dot_product_attention` appears to not work for $n > 0$, so ensure that a causal mask gets added to attn_mask or attn_bias
    if is_causal and attn_bias is None:
        attn_bias = zeros((heads, q_len, k_len), device=device, dtype=dtype)

    if attn_bias is not None:
        if attn_bias.ndim == 3:
            attn_bias = rearrange(attn_bias, 'h i j -> 1 h i j')
        attn_bias = attn_bias.expand(batch, heads, -1, -1)

        mask_value = -finfo(dtype).max

        if attn_mask is not None:
            attn_bias = attn_bias.masked_fill(~attn_mask, mask_value // 2)
        elif is_causal:
            causal_mask = _create_causal_mask(q_len, k_len, device=device)
            attn_bias = attn_bias.masked_fill(causal_mask, mask_value // 2)

        attn_mask = attn_bias

    config = _flash_attn_config(query)
    with sdp_kernel(**config._asdict()):
        return scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,  # causal mask, if requested, is included here
            dropout_p=dropout_p,
            is_causal=False  # see comment above
        )
