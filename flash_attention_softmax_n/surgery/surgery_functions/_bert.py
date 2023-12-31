from math import sqrt
from types import MethodType
from typing import Optional, Tuple

from torch import Tensor, FloatTensor, cat, matmul, tensor, long, arange, einsum
from torch.nn import Module
from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention

from flash_attention_softmax_n import softmax_n
from flash_attention_softmax_n.surgery.surgery_functions.utils import policy_registry


@policy_registry.register(BertSelfAttention, RobertaSelfAttention)
def bert_attention_converter(module: Module, module_index: int, softmax_n_param: float) -> Module:
    """Adds AttentionSoftmaxN to Bert-style SelfAttention."""
    assert isinstance(module, (BertSelfAttention, RobertaSelfAttention))
    del module_index  # unused
    module.n = softmax_n_param
    setattr(module, 'forward', MethodType(forward, module))
    return module


def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[FloatTensor] = None,
        head_mask: Optional[FloatTensor] = None,
        encoder_hidden_states: Optional[FloatTensor] = None,
        encoder_attention_mask: Optional[FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
) -> Tuple[Tensor]:
    mixed_query_layer = self.query(hidden_states)

    # If this is instantiated as a cross-attention module, the keys
    # and values come from an encoder; the attention mask needs to be
    # such that the encoder's padding tokens are not attended to.
    is_cross_attention = encoder_hidden_states is not None

    if is_cross_attention and past_key_value is not None:
        # reuse k,v, cross_attentions
        key_layer = past_key_value[0]
        value_layer = past_key_value[1]
        attention_mask = encoder_attention_mask
    elif is_cross_attention:
        key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
        value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
        attention_mask = encoder_attention_mask
    elif past_key_value is not None:
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        key_layer = cat([past_key_value[0], key_layer], dim=2)
        value_layer = cat([past_key_value[1], value_layer], dim=2)
    else:
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

    query_layer = self.transpose_for_scores(mixed_query_layer)

    use_cache = past_key_value is not None
    if self.is_decoder:
        # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
        # Additional calls to cross_attention layer can then reuse all cross-attention
        # key/value_states (first "if" case)
        # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
        # all previous decoder key/value_states. Additional calls to uni-directional self-attention
        # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
        # if encoder bidirectional self-attention `past_key_value` is always `None`
        past_key_value = (key_layer, value_layer)

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores = matmul(query_layer, key_layer.transpose(-1, -2))

    if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
        query_length, key_length = query_layer.shape[2], key_layer.shape[2]
        if use_cache:
            position_ids_l = tensor(key_length - 1, dtype=long, device=hidden_states.device).view(-1, 1)
        else:
            position_ids_l = arange(query_length, dtype=long, device=hidden_states.device).view(-1, 1)
        position_ids_r = arange(key_length, dtype=long, device=hidden_states.device).view(1, -1)
        distance = position_ids_l - position_ids_r

        positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
        positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

        if self.position_embedding_type == "relative_key":
            relative_position_scores = einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
            attention_scores = attention_scores + relative_position_scores
        elif self.position_embedding_type == "relative_key_query":
            relative_position_scores_query = einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
            relative_position_scores_key = einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
            attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

    attention_scores = attention_scores / sqrt(self.attention_head_size)
    if attention_mask is not None:
        # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
        attention_scores = attention_scores + attention_mask

    # Normalize the attention scores to probabilities.
    attention_probs = softmax_n(attention_scores, n=self.n, dim=-1)  # *** modified by CWM ***

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self.dropout(attention_probs)

    # Mask heads if we want to
    if head_mask is not None:
        attention_probs = attention_probs * head_mask

    context_layer = matmul(attention_probs, value_layer)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(new_context_layer_shape)

    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

    if self.is_decoder:
        outputs = outputs + (past_key_value,)
    return outputs
