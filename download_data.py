"""Download the Tiny Shakespeare dataset."""
import os
import requests

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "input.txt")


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    if os.path.exists(OUTPUT_PATH):
        print(f"Dataset already exists at {OUTPUT_PATH}")
        return

    print(f"Downloading Tiny Shakespeare from {DATA_URL}...")
    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(response.text)

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"Saved to {OUTPUT_PATH} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
