# MADE: Masked Autoencoder for Distribution Estimation

A from-scratch PyTorch implementation of [MADE](https://arxiv.org/abs/1502.03509) (Germain et al., ICML 2015).

![MADE Samples](made_samples.png)

## What is MADE?

MADE turns a standard autoencoder into a generative model by masking its weight matrices to enforce **autoregressive constraints**. Each output predicts the probability of a single pixel conditioned only on preceding pixels in the input ordering:

```
p(x) = p(x₁) · p(x₂|x₁) · p(x₃|x₁,x₂) · ... · p(x₇₈₄|x₁,...,x₇₈₃)
```

All 784 conditional probabilities are computed in a **single forward pass** — no sequential computation needed during training.

## How the masking works

Each hidden unit is assigned a number `m(k)` representing the maximum number of inputs it can connect to. Binary masks are constructed using two rules:

- **Input → Hidden:** allow connection if `m(k) ≥ input_index` (uses ≥)
- **Hidden → Output:** allow connection if `output_index > m(k)` (uses strict >)

These two rules guarantee that any path from input `j` to output `i` requires `i > j` — the autoregressive property. The asymmetry (≥ vs >) is critical: using strict > everywhere would kill most connections and starve the network of capacity, while using ≥ everywhere would let outputs see their own input.

## Implementation details

- **MaskedLinear layers** with binary masks applied via element-wise multiplication on weights
- Single hidden layer, 500 units, ReLU activation
- Trained on binarized MNIST (thresholded at 0.5) with binary cross-entropy loss
- Adam optimizer, lr=0.001, 30 epochs
- Autoregressive sampling: 784 sequential forward passes, one per pixel

## Project structure

```
├── MADE_1.py          # Full implementation: model, training, sampling
├── made_samples.png   # Generated digit samples
└── README.md
```

## Usage

```bash
pip install torch torchvision matplotlib
python MADE_1.py
```

Training takes a few minutes on CPU. After training, 16 digit samples are generated and saved to `made_samples.png`.

## Results

The model generates recognizable handwritten digits after 30 epochs of training. Samples are noisy but clearly digit-like — consistent with the paper's results for a single hidden layer MADE.

## Verified properties

- **Autoregressive property test:** for every output `i` and every input `j ≥ i`, perturbing input `j` produces zero change in output `i`
- **Sample quality:** generated images resemble handwritten digits comparable to Figure 3 in the original paper

## Reference

Germain, M., Gregor, K., Murray, I., & Larochelle, H. (2015). *MADE: Masked Autoencoder for Distribution Estimation.* Proceedings of the 32nd International Conference on Machine Learning (ICML).
