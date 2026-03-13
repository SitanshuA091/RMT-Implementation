# Recurrent Memory Transformer Experiments

This project explores the implementation and evaluation of the Recurrent Memory Transformer (RMT), a memory-augmented extension of the Transformer architecture proposed by Yandex Research. The model introduces segment-level recurrence using special memory tokens that pass information between sequence segments, enabling the model to process much longer contexts than standard Transformers.

## Objectives

* Implement a baseline decoder-only Transformer language model in PyTorch.
* Implement a simplified RMT-style memory mechanism.
* Evaluate the effect of memory on long-context language modeling.

## Experiments

The project includes two main experimental setups:

1. **Baseline Transformer**

   * Train a small decoder-only Transformer from scratch.
   * Evaluate language modeling performance on long sequences.

2. **Augmented Transformer with Memory Tokens**

   * Introduce memory tokens passed between sequence segments.
   * Compare performance with the baseline model.

## Datasets

Experiments will primarily use:

* WikiText-103

## Evaluation

Models will be evaluated using:

* Perplexity for language modeling
* BLEU / ROUGE for generation tasks
* Long-range dependency tests

## Added Optimizations for Training based on paper
- Mixed Precision Training (FP16 + AMP):
Mixed precision uses lower-precision (FP16) arithmetic for most tensor operations while keeping critical calculations in FP32. This reduces GPU memory usage and speeds up matrix computations.
- Gradient Accumulation:
Gradient accumulation simulates a larger batch size by accumulating gradients over multiple forward/backward passes before performing an optimizer update. This is useful when GPU memory cannot support large batches,
