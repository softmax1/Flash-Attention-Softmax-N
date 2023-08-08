# FlashAttention-with-Softmax1

Using the Triton language to implement Flash Attention with Softmax_1. 
Flash Attention computes the numerator and denominator of Attention separately, so all we need to do is add the "+1" term to the denominator.
Note however that the +1 needs to be "shifted," see [#10](https://github.com/softmax1/softmax1/issues/10).

## Usage
The main function is 
```python
from src.flash_attn_triton_softmax1 import attention
```
Here are the signatures of the forward and backward methods
```python
def forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False, sm_scale: Optional[float] = None) -> torch.Tensor:
    """
    Triton implementation of forward pass of Flash Attention with Softmax_1

    :param q: Query tensor; shape (N, ..., L, E).
    :param k: Key tensor; shape (N, ..., S, E).
    :param v: Value tensor; shape (N, ..., S, Ev).
    :param causal: If true, assumes causal attention masking.
    :param sm_scale: Scaling factor applied prior to softmax. If None, the default value is set to 1 / sqrt(E).
    :return: Attention output; shape (N, ..., L, Ev).
    """

def backward(do: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
    """
    Triton implementation of backward pass of Flash Attention with Softmax_1
    :param do: Output Gradient tensor; shape (N, ..., L, Ev). $\partial \phi / \partial \vec{o}$ where $\phi$ is the loss function
    :return: Gradients of the Query, Key, and Value tensors along with two null values.
    """
```

Shapes:
- N: batch size. (N, ...) indicates the effective batch size could span multiple dimensions, e.g. the batch size and the number of attention heads
- S: source sequence length
- L: target sequence length
- E: embedding dimension of the query and key
- Ev: embedding dimension of the value

Gradients are of the loss function, $\phi$, with respect to a tensor. For example, given $t_{ijk}$:
$$(\partial t)_{ijk} \equiv \frac{\partial \phi}{\partial t^{ijk}}$$.

## Limitations
- Currently, no Triton implementation of Flash Attention, here or elsewhere, has dropout. This is a known limitation of using Triton versus CUDA.
- This implementation only works with `dtype in {torch.float16, torch.bfloat16}` for the query, key, and value tensors.
- This implementation also expects there to be multiple attention heads. That is, the query, key, and value tensors must be 4-dimensional. This should probably be generalizable as well.
- I started with $n = 1$, but it should be easy generalize to $n \in \mathbb{R}$.

## Testing
The Triton language is not available on CPUs.
Therefore, we need to use a GPU to fully test the implementation.

My testing used the following versions:
```
triton==2.0.0.post1
triton-nightly==2.1.0.dev20230808020556
```

## Links
- [Flash Attention paper](https://arxiv.org/abs/2205.14135)
- [Code associated with paper](https://github.com/Dao-AILab/flash-attention/tree/main)
- [Triton documenation](https://triton-lang.org/main/index.html)
- [Triton Attention tutorial](https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py); This is the source for the "OG" code.
- [MoasicBERT](https://github.com/mosaicml/examples/tree/845bfe23c77316264d5dd6e2a6b7c46cefa4519a/examples/benchmarks/bert); Benchmark for training BERT cheaply in part using Flash Attention

## Contribute
Feel free to suggest extensions, additional tests, etc. by raising an issue.
