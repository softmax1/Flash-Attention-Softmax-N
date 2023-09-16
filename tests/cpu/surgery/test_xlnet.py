from pytest import raises
from torch import randn
from torch.testing import assert_close
from transformers import AutoModel

from flash_attention_softmax_n.surgery import apply_attention_softmax_n
from tests.common import device_name


MODEL_NAME = 'hf-internal-testing/tiny-random-XLNetModel'  # a tiny XLNet for testing purposes
INPUT_TENSOR_SIZE = (2, 3, 32)  # the size of the input embedding to XLNet


def test_xlnet(device_name):
    # load the tiny XLNet
    original_model = AutoModel.from_pretrained(MODEL_NAME)

    # load the same XLNet and "operate" on its `rel_attn_core` method, but continue to use softmax_0
    new_model_0 = AutoModel.from_pretrained(MODEL_NAME)
    apply_attention_softmax_n(model=new_model_0, softmax_n_param=0.)

    for layer_idx in range(new_model_0.config.num_hidden_layers):
        # new model should have `n` parameter with `n == 0`
        assert new_model_0.layer[layer_idx].rel_attn.n == 0.

        # original model should not have `n` parameter
        with raises(AttributeError):
            original_model.layer[layer_idx].rel_attn.n

        inputs_embeds = randn(size=INPUT_TENSOR_SIZE)
        original_ouptut = original_model(inputs_embeds=inputs_embeds)
        new_output_0 = new_model_0(inputs_embeds=inputs_embeds)
        # new model with `n == 0` should produce the same output as the original model
        assert_close(new_output_0, original_ouptut)

    # load the same XLNet and "operate" on its `rel_attn_core` method, and now use softmax_1
    new_model_1 = AutoModel.from_pretrained(MODEL_NAME)
    apply_attention_softmax_n(model=new_model_1, softmax_n_param=1.)

    for layer_idx in range(new_model_1.config.num_hidden_layers):
        # new model should have `n` paramter with `n == 1`
        assert new_model_1.layer[layer_idx].rel_attn.n == 1.

        inputs_embeds = randn(size=INPUT_TENSOR_SIZE)
        new_output_1 = new_model_1(inputs_embeds=inputs_embeds)
        new_output_0 = new_model_0(inputs_embeds=inputs_embeds)
        # the output with `n == 1` should be different than when `n == 0`
        assert new_output_1[0].sum() != new_output_0[0].sum()
