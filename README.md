# nanoGPT — Shakespeare

A from-scratch implementation of a small GPT-style transformer, trained to generate Shakespeare-like text at the character level.

This project follows Andrej Karpathy's ["Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY) tutorial, with the goal of understanding every line of a working transformer.

## Goals

- Implement a decoder-only transformer from scratch in PyTorch
- Train it on the Tiny Shakespeare dataset
- Generate plausible-looking Shakespearean text
- Understand attention, multi-head attention, positional embeddings, residual connections, and layer norm well enough to explain them

## Project structure

```
nanogpt-shakespeare/
├── data/              # Dataset (input.txt) — gitignored
├── checkpoints/       # Saved model weights — gitignored
├── notes/             # Personal notes, observations, debugging logs
├── model.py           # The GPT model definition
├── train.py           # Training loop
├── sample.py          # Generate text from a trained model
├── download_data.py   # Fetch the Tiny Shakespeare dataset
├── requirements.txt
└── README.md
```

## Setup

```bash
# Clone the repo
git clone https://github.com/<your-username>/nanogpt-shakespeare.git
cd nanogpt-shakespeare

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the dataset
python download_data.py
```

## Usage

```bash
# Train the model
python train.py

# Generate text from the trained model
python sample.py
```

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

(I'll fill this in as I go — see `notes/` for the running log.)

## Credits

- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) — the original
- ["Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY) — the tutorial this follows
- ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) — the original transformer paper
