# miniGPT — Shakespeare

A from-scratch, character-level GPT trained to generate Shakespeare-like text, built by working up from a bigram baseline to a full decoder-only transformer. The project follows Andrej Karpathy's ["Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY), with the goal of understanding every component of a working transformer rather than calling into a library.

Everything operates at the character level on the Tiny Shakespeare corpus — the vocabulary is just the 65 unique characters in the text, so there's no external tokenizer; the model learns to predict the next character.

## Two models

The repo is structured as a learning progression, from the simplest possible baseline to the full architecture:

- **`bigram.py`** — a bigram baseline. Each token looks up next-token logits through an embedding table and a linear head, with no notion of context beyond the current character. Trains in seconds and sets the loss floor to beat.
- **`gpt.py`** — the full decoder-only transformer: token + positional embeddings, stacked pre-norm transformer blocks (multi-head causal self-attention + feed-forward), residual connections, layer norm, and dropout.

## What's implemented

- [x] Bigram language model baseline
- [x] Self-attention (single head) with causal masking
- [x] Multi-head attention
- [x] Feed-forward layers and residual connections
- [x] Layer norm (pre-norm placement)
- [x] Full transformer block, stacked
- [x] Dropout and regularization

See [`learning_log.md`](learning_log.md) for running notes on the concepts — causal masking via `tril`, why logits are reshaped to `(B*T, C)` for cross-entropy, and how the attention output projection lets heads combine information.

## Architecture (`gpt.py`)

- **Attention** — each `Head` computes scaled dot-product attention with a lower-triangular causal mask; `MultiHeadAttention` runs several heads in parallel, concatenates them, and applies an output projection.
- **Blocks** — pre-norm residual blocks: `x = x + attn(ln1(x))` then `x = x + ffwd(ln2(x))`, with a `Linear → ReLU → Linear` feed-forward at 4× expansion.
- **Model** — summed token and positional embeddings, a stack of blocks, a final layer norm, and a linear head projecting to vocab logits.
- **Generation** — autoregressive sampling that crops context to `block_size`, softmaxes the final-position logits, and samples with `torch.multinomial`.

## Configuration

| | Bigram | GPT |
|---|---|---|
| Parameters | ~2K | **~3.22M** |
| Context (`block_size`) | 8 | 128 |
| Embedding dim | 32 | 256 |
| Layers | — | 4 |
| Heads | — | 4 (head size 64) |
| Dropout | — | 0.2 |
| Learning rate | 1e-2 | 3e-4 |
| Iterations | 3,000 | 3,000 |
| Optimizer | AdamW | AdamW |

Vocab size is 65 (unique characters). The GPT auto-selects Apple Silicon `mps` and falls back to `cpu`.

## Results

The model is trained with a standard next-token cross-entropy objective. As a reference point, a model that predicted characters uniformly at random would have a loss of `ln(65) ≈ 4.17`; training drives this well below that as the model learns character statistics, common words, and eventually Shakespearean structure (speaker labels, line breaks, archaic diction).

> **Final train/val loss:** _add your reported values here._ The training loop prints train/val loss every 300 steps — capturing the final numbers (and committing a sample generation) gives the repo a concrete result.

## Setup

```bash
pip install torch
```

The Tiny Shakespeare corpus (`input.txt`) is included, so no download is needed.

## Usage

Run the bigram baseline:

```bash
python bigram.py
```

Train the full GPT and print a 500-character sample:

```bash
python gpt.py
```

Both scripts print train/val loss periodically and generate a text sample at the end.

## Credits

- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) — the original
- ["Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY) — the tutorial this follows
- ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) — the original transformer paper
