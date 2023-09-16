from types import MethodType

from torch import einsum, float16
from torch.nn import Module
from transformers.models.xlnet.modeling_xlnet import XLNetRelativeAttention

from flash_attention_softmax_n import softmax_n
from flash_attention_softmax_n.surgery.surgery_functions.utils import policy_registry


@policy_registry.register(XLNetRelativeAttention)
def xlnet_attention_converter(module: Module, module_index: int, softmax_n_param: float) -> Module:
    """Adds AttentionSoftmaxN to XLNet RelativeAttention."""
    assert isinstance(module, (XLNetRelativeAttention,))
    del module_index  # unused

    if softmax_n_param < 0.:
        raise ValueError(f"Softmax `n` parameter must be non-negative, found n={softmax_n_param}.")
    module.n = softmax_n_param

    setattr(module, 'rel_attn_core', MethodType(rel_attn_core, module))
    return module


def rel_attn_core(
    self,
    q_head,
    k_head_h,
    v_head_h,
    k_head_r,
    seg_mat=None,
    attn_mask=None,
    head_mask=None,
    output_attentions=False,
):
    """Core relative positional attention operations."""

    # content based attention score
    ac = einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)

    # position based attention score
    bd = einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    bd = self.rel_shift_bnij(bd, klen=ac.shape[3])

    # segment based attention score
    if seg_mat is None:
        ef = 0
    else:
        ef = einsum("ibnd,snd->ibns", q_head + self.r_s_bias, self.seg_embed)
        ef = einsum("ijbs,ibns->bnij", seg_mat, ef)

    # merge attention scores and perform masking
    attn_score = (ac + bd + ef) * self.scale
    if attn_mask is not None:
        # attn_score = attn_score * (1 - attn_mask) - 1e30 * attn_mask
        if attn_mask.dtype == float16:
            attn_score = attn_score - 65500 * einsum("ijbn->bnij", attn_mask)
        else:
            attn_score = attn_score - 1e30 * einsum("ijbn->bnij", attn_mask)

    # attention probability
    attn_prob = softmax_n(attn_score, n=self.n, dim=3)  # *** modified by CWM ***
    attn_prob = self.dropout(attn_prob)

    # Mask heads if we want to
    if head_mask is not None:
        attn_prob = attn_prob * einsum("ijbn->bnij", head_mask)

    # attention output
    attn_vec = einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)

    if output_attentions:
        return attn_vec, einsum("bnij->ijbn", attn_prob)

    return attn_vec
