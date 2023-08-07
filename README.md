# FlashAttention-with-Softmax1

Using the Triton language to try to implement Flash Attention with Softmax_1. 
Flash Attention computes the numerator and denominator of Attention separately, so all we need to do is add the "+1" term to the denominator.
Note however that the +1 needs to be "shifted," see [#10](https://github.com/softmax1/softmax1/issues/10).

## Testing
The Triton language is not available on CPUs.
Therefore, we need to use a GPU to fully test the implementation.
So far the CPU tests have passed on my Macbook Air.
I haven't had a chance to run this on AWS yet.

## Links
- [Flash Attention paper](https://arxiv.org/abs/2205.14135)
- [Code associated with paper](https://github.com/Dao-AILab/flash-attention/tree/main)
- [Triton Attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)