# FlashAttention-with-Softmax1

Using the Triton language to try to implement Flash Attention with Softmax_1. 
Flash Attention computes the numerator and denominator of Attention separately, so all we need to do is add the "+1" term to the denominator.
Note however that the +1 needs to be "shifted," see [#10](https://github.com/softmax1/softmax1/issues/10).

## Usage
I'll complete this if/when I get something implemented.

## Limitations
- Currently, no Triton implementation of Flash Attention, here or elsewhere, has dropout. This is a known limitation of using Triton versus CUDA.
- This implementation only works with `dtype=torch.float16` for the query, key, and value tensors. I'd naively guess it should straightforward to generalize this.
- This implementation also expects there to be multiple attention heads. That is, the query, key, and value tensors must be 4-dimensional. This should probably be generalizable as well.

## Testing
The Triton language is not available on CPUs.
Therefore, we need to use a GPU to fully test the implementation.

The version I tested was:
```
triton==2.0.0.post1
triton-nightly==2.1.0.dev20230808020556
```

## Links
- [Flash Attention paper](https://arxiv.org/abs/2205.14135)
- [Code associated with paper](https://github.com/Dao-AILab/flash-attention/tree/main)
- [Triton Attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)

## Contribute
Feel free to suggest other tests to perform by raising an issue.
