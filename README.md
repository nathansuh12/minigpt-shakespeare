# miniGPT — Shakespeare

A from-scratch implementation of a small GPT-style transformer, trained to generate Shakespeare-like text at the character level.

This project follows Andrej Karpathy's ["Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY) tutorial, with the goal of understanding every line of a working transformer.

## Goals

- Implement a decoder-only transformer from scratch in PyTorch
- Train it on the Tiny Shakespeare dataset
- Generate plausible-looking Shakespearean text
- Understand attention, multi-head attention, positional embeddings, residual connections, and layer norm well enough to explain them


## Progress / Learning log

I'm committing at natural checkpoints to track my learning progression:

- [ ] Bigram language model baseline
- [ ] Self-attention (single head)
- [ ] Multi-head attention
- [ ] Feedforward layers and residual connections
- [ ] Layer norm
- [ ] Full transformer block, stacked
- [ ] Dropout and regularization
- [ ] Training run that generates coherent-ish Shakespeare

## What I learned

see learning_log.md

## Credits

- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) — the original
- ["Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY) — the tutorial this follows
- ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) — the original transformer paper
