"""Train the model on Tiny Shakespeare."""
import os
import torch
from model import BigramLanguageModel

# ---- Hyperparameters ----
BATCH_SIZE = 32           # how many independent sequences per batch
BLOCK_SIZE = 8            # context length (will grow to 256 by end of video)
MAX_ITERS = 3000
EVAL_INTERVAL = 300
LEARNING_RATE = 1e-2      # higher early; lower to ~3e-4 once attention is added
EVAL_ITERS = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------------------------------

torch.manual_seed(1337)

# Load data
data_path = os.path.join(os.path.dirname(__file__), "data", "input.txt")
with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

# Build a character-level vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# Train / val split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([d[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    print(f"Vocab size: {vocab_size}")
    print(f"Training on {DEVICE}")

    model = BigramLanguageModel(vocab_size).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for it in range(MAX_ITERS):
        if it % EVAL_INTERVAL == 0 or it == MAX_ITERS - 1:
            losses = estimate_loss(model)
            print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", "model.pt")
    torch.save({"model_state": model.state_dict(), "vocab_size": vocab_size,
                "stoi": stoi, "itos": itos}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    # Quick generation sample
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    print("\n--- Sample ---")
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))


if __name__ == "__main__":
    main()
