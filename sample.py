"""Generate text from a trained model checkpoint."""
import os
import torch
from model import BigramLanguageModel

CHECKPOINT_PATH = os.path.join("checkpoints", "model.pt")
MAX_NEW_TOKENS = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"No checkpoint at {CHECKPOINT_PATH}. Run train.py first."
        )

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    vocab_size = ckpt["vocab_size"]
    itos = ckpt["itos"]
    decode = lambda l: "".join([itos[i] for i in l])

    model = BigramLanguageModel(vocab_size).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    output = model.generate(context, max_new_tokens=MAX_NEW_TOKENS)[0].tolist()
    print(decode(output))


if __name__ == "__main__":
    main()
