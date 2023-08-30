from pytest import mark
from torch import randn
from torch.nn import Module, Linear

from flash_attention_softmax_n import flash_attention_n
from flash_attention_softmax_n.analysis.hooks import register_activation_hooks, compute_weight_statistics


class Transformer(Module):
    def __init__(self, return_inputs: bool = False):
        super().__init__()
        self.return_inputs = return_inputs
        self.linear1 = Linear(32, 16)
        self.linear2 = Linear(16, 7)

    def forward(self, q, k, v):
        fa = flash_attention_n(q, k, v)
        l1 = self.linear1(fa)
        l2 = self.linear2(l1)
        return l2, q, k, v if self.return_inputs else l2


@mark.parametrize("acts_to_save", [None, "linear1,linear2"])
def test_register_activation_hooks(acts_to_save):
    model = Transformer()
    to_save = None if acts_to_save is None else set(acts_to_save.split(','))

    # register fwd hooks in specified layers
    saved_activations = register_activation_hooks(model, layers_to_save=to_save)

    # run twice, then assert each created lists for module, each with length n_batches * batch_size
    n_batches = 2
    batch_size = 10
    for _ in range(n_batches):
        query = randn((batch_size, 1, 1024, 64))
        key = randn((batch_size, 1, 1152, 64))
        value = randn((batch_size, 1, 1152, 32))

        model(query, key, value)

    print(saved_activations)

    assert len(saved_activations) == 0 if to_save is None else len(to_save)

    for activation in saved_activations:
        assert len(saved_activations[activation]) == 5
        assert saved_activations[activation]['n_samples'] == n_batches * batch_size
        assert saved_activations[activation]['kurtosis']**2 >= 0.
        assert saved_activations[activation]['skewness']**2 >= 0.
        assert saved_activations[activation]['variance']**2 >= 0.
        assert saved_activations[activation]['mean']**2 >= 0.


def test_compute_weight_statistics():
    model = Transformer()

    n_batches = 1
    batch_size = 6
    for _ in range(n_batches):
        query = randn((batch_size, 3, 1024, 32))
        key = randn((batch_size, 1024, 32))
        value = randn((batch_size, 1024, 32))

        model(query, key, value)

    weight_stats = compute_weight_statistics(model)
    for name in weight_stats:
        assert len(weight_stats[name]) == 5
    assert weight_stats['linear1.weight']['n_weights'] == 32 * 16
    assert weight_stats['linear1.bias']['n_weights'] == 16
    assert weight_stats['linear2.weight']['n_weights'] == 16 * 7
    assert weight_stats['linear2.bias']['n_weights'] == 7
