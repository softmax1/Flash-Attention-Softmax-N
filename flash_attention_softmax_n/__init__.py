from warnings import warn

from einops._torch_specific import allow_ops_in_compiled_graph

from flash_attention_softmax_n.flash_attn import flash_attention_n
from flash_attention_softmax_n.functional import softmax_n, slow_attention_n
try:
    from flash_attention_softmax_n.flash_attn_triton import flash_attention_n_triton
except ModuleNotFoundError as e:
    warn(f'The Triton flash attention implementation, `flash_attention_n_triton`, is not available. {e}.')
    flash_attention_n_triton = None


allow_ops_in_compiled_graph()
