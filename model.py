"""
GPT model definition.

Start here with a simple bigram model, then progressively add components
as you follow along with Karpathy's video:
  1. Bigram language model (current)
  2. Self-attention (single head)
  3. Multi-head attention
  4. Feedforward + residual connections
  5. Layer norm
  6. Full transformer block (stacked)
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    """A simple bigram language model — the starting point.

    Each token directly predicts the next via an embedding table.
    No attention, no context beyond the previous token.
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensors of integers
        logits = self.token_embedding_table(idx)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Generate text by sampling one token at a time."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # take the last time step
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# TODO: as you progress through the video, build up to a full GPT class:
#
# class GPT(nn.Module):
#     def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
#         super().__init__()
#         self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
#         self.position_embedding_table = nn.Embedding(block_size, n_embd)
#         self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
#         self.ln_f = nn.LayerNorm(n_embd)
#         self.lm_head = nn.Linear(n_embd, vocab_size)
#         ...
