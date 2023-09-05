from pytest import raises
from torch import randn
from torch.testing import assert_close
from transformers import AutoModel

from flash_attention_softmax_n.surgery import apply_attention_softmax_n
from tests.common import device_name


def test_bert(device_name):
    model_name = 'prajjwal1/bert-tiny'
    original_model = AutoModel.from_pretrained(model_name)

    new_model_0 = AutoModel.from_pretrained(model_name)
    apply_attention_softmax_n(model=new_model_0, softmax_n_param=0.)

    for layer_idx in range(new_model_0.config.num_hidden_layers):
        assert new_model_0.encoder.layer[layer_idx].attention.self.n == 0.

        with raises(AttributeError):
            assert original_model.encoder.layer[layer_idx].attention.self.n == 0.

        hidden_states = randn((2, 3, new_model_0.config.hidden_size), device=device_name)
        original_ouptut = original_model.encoder.layer[layer_idx].attention.self(hidden_states)
        new_output_0 = new_model_0.encoder.layer[layer_idx].attention.self(hidden_states)
        assert_close(new_output_0, original_ouptut)

    new_model_1 = AutoModel.from_pretrained(model_name)
    apply_attention_softmax_n(model=new_model_1, softmax_n_param=1.)

    for layer_idx in range(new_model_1.config.num_hidden_layers):
        assert new_model_1.encoder.layer[layer_idx].attention.self.n == 1.

        hidden_states = randn((2, 3, new_model_1.config.hidden_size), device=device_name)
        new_output_1 = new_model_1.encoder.layer[layer_idx].attention.self(hidden_states)
        new_output_0 = new_model_0.encoder.layer[layer_idx].attention.self(hidden_states)
        assert new_output_1[0].sum() != new_output_0[0].sum()
