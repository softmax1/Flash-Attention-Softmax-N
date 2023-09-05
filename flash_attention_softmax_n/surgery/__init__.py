try:
    from flash_attention_softmax_n.surgery.attention_softmax_n import AttentionSoftmaxN, apply_attention_softmax_n
    SURGERY_INSTALLED = True
except ModuleNotFoundError:
    SURGERY_INSTALLED = False
