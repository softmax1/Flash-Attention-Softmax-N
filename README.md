# Flash-Attention-Softmax-N

[Flash attention](https://arxiv.org/abs/2205.14135) with softmaxN.
[Attention is Off By One](https://www.evanmiller.org/attention-is-off-by-one.html) hypothesized that using softmax1 in the attention mechanism will reduce the number of outliers in the activations and weights of a transformer model.

🎯**Efficent, Numerically-Stable Implementation of SoftmaxN**: No more worrying about the non-trivial implementation of softmaxN.
$$\text{softmax}_n(x_i) = \frac{\exp(x_i)}{n + \sum_j \exp(x_j)}$$

🚀 **Multiple Attention Implementations, your choice**: Whatever you're aiming for, we've got you covered with three Attention implementations.
In the spirit of the flash attention paper, further gains can be made by considering the whole attention function instead of just the softmaxN subfunction.
- `flash_attention_n`: recommended for integer values of _n_, uses CUDA on the backend if a GPU is available 
- `flash_attention_n_triton`: recommended for non-integer values of _n_ when a GPU is available, uses Triton
- `slow_attention_n`: flexible, torch-based implementation

🧠 **Run statistical analyses**: Compute summary statistics for both the weights and activations of your model.
The activation stats are computed online as the model is training.

🔥 **Perform surgery on existing models** Take a pretrained model with softmax_0 in its attention mechanism and "operator" on it to replace softmax_0 with softmax_n.

## Install
Simple installation
```bash
$ pip install flash-attention-softmax-n
```
Optionally install the Triton implementation
```bash
$ pip install flash-attention-softmax-n[triton]
$ pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
```

## Usage


|              Feature / Function              | `flash_attention_n` |    `flash_attention_n_triton`    | `slow_attention_n` |
|:--------------------------------------------:|:-------------------:|:--------------------------------:|:------------------:|
|               CPU-compatible?                |         Yes         |                No                |        Yes         |
|          Real or Integer valued $n$          |       Integer       |               Real               |        Real        |
|    Datatype(s) natively supported on GPU     |  fp32, fp16, bf16   |        fp16 (*see below)         |  fp32, fp16, bf16  |
|     Datatypes natively supported on CPU      |     fp32, bf16      |               n/a                |     fp32, bf16     |
|                   Dropout?                   |         Yes         |                No                |        Yes         |
|                 Causal Mask?                 |         Yes         | only tested for $n \leq 10^{-3}$ |        Yes         |
|            Attention Bias (ALiBi)            |         Yes         |                No                |         No         |
|                Attention Mask                |         Yes         |                No                |        Yes         |
|          supports `query.ndim < 4`           |         No          |                No                |        Yes         |
| supports `key.ndim < 4` and `value.ndim < 4` |         Yes         |                No                |        Yes         |
| requries `key.shape[-1] == value.shape[-1]`  |         No          |               Yes                |         No         |

### CUDA
The recommendation function to use for integer-values of _n_ with or without a GPU.
You'll probably need an A100 to reap the full benefit though.
This implementation was inspired by [x-transformers](https://github.com/lucidrains/x-transformers/tree/main).
It uses `torch.nn.functional.scaled_dot_product_attention` on the backend, which requires `torch>=2.0.0`.

```python
import torch
from flash_attention_softmax_n import flash_attention_n

softmax_n_param = 1
query = torch.randn((6, 1, 1024, 64))
key = torch.randn((6, 1152, 64))
value = torch.randn((6, 1152, 32))

attn = flash_attention_n(
    query=query,
    key=key,
    value=value,
    softmax_n_param=softmax_n_param,
    scale=None,
    dropout_p=0.,
    attn_mask=None,
    attn_bias=None,
    is_causal=False
)
```

### Triton
The recommended function to use when you want GPU acceleration and have a non-integer-valued _n_.
Note the Triton implementation has a more limited set of features compared to the CUDA version, see the above comparison table.
*To use datatypes other than `fp16` first convert your input to `fp16` and then convert the attention output back to your original datatype.
This is a generalization of OpenAI's Triton fused attention [implementation](https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py).
Requires `torch>=2.0.0` and `triton>=2.0.0`.

```python
import torch
from flash_attention_softmax_n import flash_attention_n_triton

softmax_n_param = 1.
query = torch.randn((6, 1, 1024, 64))
key = torch.randn((6, 1, 1152, 64))
value = torch.randn((6, 1, 1152, 64))

attn = flash_attention_n_triton(
    query=query,
    key=key,
    value=value,
    softmax_n_param=softmax_n_param,
    scale=None,
    is_causal=False
)
```

### Slow Attention
Written in torch.
Use this version when you have a real-valued _n_, and the Triton version is unavailable or doesn't have the feature(s) you need.

```python
import torch
from flash_attention_softmax_n import slow_attention_n

softmax_n_param = 1.
query = torch.randn((6, 1024, 64))
key = torch.randn((6, 1152, 64))
value = torch.randn((6, 1152, 32))

attn = slow_attention_n(
    query=query,
    key=key,
    value=value,
    softmax_n_param=softmax_n_param,
    scale=None,
    dropout_p=0.,
    attn_mask=None,
    is_causal=False,
    softmax_dtype=None,
    train=True
)
```

We also provide a torch implementation of softmaxN that can be used as a drop-in replacement for softmax.
```python
import torch
from flash_attention_softmax_n import softmax_n

x = torch.rand((100, 100))
# y = torch.nn.functional.softmax(x, dim=-1, dtype=torch.float32)
y = softmax_n(x, dim=-1, dtype=torch.float32)

y1 = softmax_n(x, n=1.)
```

### Statistical Analysis
```python
from flash_attention_softmax_n.analysis import register_activation_hooks, compute_weight_statistics, save_results

model = GPT4()  # XD
activations_statistics = register_activation_hooks(model)  # activation stats are computed online during training, so register the hooks in advance

trainer.train(model)

weight_statistics = compute_weight_statistics(model)  # weights stats are coputed after training is finished

print(activations_statistics['...attention.output...']['kurtosis'])
print(weight_statistics['...attention.output...']['kurtosis'])

save_results({'activations': activations_statistics, 'weights': weight_statistics}, 'my-gpt4')
```

### Surgery
Functional API: add one line of code to your script.
```python
import transformers

from flash_attention_softmax_n.surgery import apply_attention_softmax_n


model = transformers.AutoModel.from_pretrained('bert-base-uncased')
apply_attention_softmax_n(model=model, softmax_n_param=1.)
...
```

Object-oriented API for use with the MosaicML composer trainer.
```python
import composer
import transformers

from flash_attention_softmax_n.surgery import AttentionSoftmaxN


model = transformers.AutoModel.from_pretrained('bert-base-uncased')
trainer = composer.trainer.Trainer(
    model=model,
    algorithms=[AttentionSoftmaxN(softmax_n_param=1.)]
)
...
```

Add your model to the registry!
(Currently, only BERT and RoBERTa without flash attention are available by default.)
As an example, use `policy_registry` to replace slow_attention_0 in `MyModel` with flash_attention_n.
After registration, wrap the model in `apply_attention_softmax_n`.
```python
import types

import torch

from flash_attention_n import slow_attention_n, flash_attention_n
from flash_attention_softmax_n.surgery import apply_attention_softmax_n
from flash_attention_n.surgery.surgery_functions import policy_registry


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = SlowAttention()

    def forward(self, q, k, v):
        return self.attn(q, k, v, softmax_n_param=0.)


class SlowAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        return slow_attention_n(q, k, v, softmax_n_param=0.)


@policy_registry.register(SlowAttention)
def slow_attention_converter(module: torch.nn.Module, module_index: int, softmax_n_param: float) -> torch.nn.Module:
    assert isinstance(module, SlowAttention)
    del module_index  # unused
    module.n = softmax_n_param
    setattr(module, 'forward', types.MethodType(forward, module))
    return module


def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return flash_attention_n(q, k, v, softmax_n_param=int(self.n))


if __name__ == '__main__':
    model = MyModel()
    apply_attention_softmax_n(model=model, softmax_n_param=1.)  # will log a warning if the model isn't registered
```