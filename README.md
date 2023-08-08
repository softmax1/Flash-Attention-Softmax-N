# FlashAttention-with-Softmax1

Using the Triton language to try to implement Flash Attention with Softmax_1. 
Flash Attention computes the numerator and denominator of Attention separately, so all we need to do is add the "+1" term to the denominator.
Note however that the +1 needs to be "shifted," see [#10](https://github.com/softmax1/softmax1/issues/10).

## Strategy
The `_fwd_kernel` returns the output, $O$, the denominator, $\ell$, and the max value(s) of the inputs, $m$.
As such, the initial strategy is to weight the output
$$O \to O^\prime = O \cdot \frac{\ell}{\ell + e^{-m}} .$$

## Usage
I'll complete this if/when I get something implemented.

## Limitations
- No Triton implementation of Flash Attention, here or elsewhere, has dropout. Currently, this is a known limitation of using Triton versus CUDA.
- I'm starting with "the OG" Triton implementation of Flash Attention, which also does not support causal masking. If this goes well, I try to upgrade to v2, which does allow for causal masking.

## Testing
The Triton language is not available on CPUs.
Therefore, we need to use a GPU to fully test the implementation.
So far the CPU tests have passed on my Macbook Air.
I haven't had a chance to run this on AWS yet.

## Links
- [Flash Attention paper](https://arxiv.org/abs/2205.14135)
- [Code associated with paper](https://github.com/Dao-AILab/flash-attention/tree/main)
- [Triton Attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)

## Contribute
Feel free to suggest other tests to perform by raising an issue.
