# Panel A — RQVI Architecture & Inference

Two schematic diagrams illustrating the Residual-Quantized Variational Inference (RQVI) model architecture and its post-training inference procedure.

## Sub-panels

### Top — Model Architecture (RQVI_architecture.png)

Residual-Quantized VAE architecture overview:

- **Encoder** — scVI-based encoder maps input gene expression to a continuous latent space
- **Residual Quantization** — N codebooks with residual connections decompose the latent representation from coarse to fine gene programs; each codebook quantizes the residual left by previous codebooks
- **Quantized Latent** — codebook embeddings are summed into the quantized latent Z_q
- **Decoder** — scVI-based decoder reconstructs gene expression from Z_q
- **Training Objective** — reconstruction loss + VQ commitment loss

### Bottom — Post-Training Inference (RQVI_training.png)

Inference procedure producing two output matrices:

- **GP Loading Matrix** — cell × GP loading scores computed via distance-based softmax activation: `s_i = -d_i / τ`, where `d_i` is the distance from the encoder output to codebook entry `i` and `τ` is a temperature parameter
- **Gene Weights Matrix** — GP × gene effect weights computed via differential log-expression: the difference in decoder log-expression between a (GP + baseline) forward pass and a baseline-only forward pass, isolating each GP's contribution to gene expression

## Source figures

| File | Description |
|------|-------------|
| `figures/RQVI_figure_method.pdf` | Combined method schematic (architecture + inference) |

## How to reproduce

These are hand-drawn schematic diagrams — no script or data is needed to reproduce them.
