from einops._torch_specific import allow_ops_in_compiled_graph

from flash_attention_softmax_n.core.flash_attn import flash_attention_n
from flash_attention_softmax_n.core.functional import softmax_n, slow_attention_n
try:
    from flash_attention_softmax_n.core.flash_attn_triton import flash_attention_n_triton
    TRITON_INSTALLED = True
except ModuleNotFoundError:
    TRITON_INSTALLED = False


allow_ops_in_compiled_graph()
