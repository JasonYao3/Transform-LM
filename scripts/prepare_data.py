import os
import sys
import json
import torch
import numpy as np
from transformer_lm.tokenizer import BPETokenizer
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode


def load_tokenizer():
    # Reuse loading logic
    vocab_path = FIXTURES_PATH / "gpt2_vocab.json"
    merges_path = FIXTURES_PATH / "gpt2_merges.txt"

    with open(vocab_path) as f:
        gpt2_vocab = json.load(f)

    merges = []
    with open(merges_path) as f:
        for line in f:
            line = line.strip()
            if line and len(line.split()) == 2:
                merges.append(tuple(line.split()))

    byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    vocab = {
        idx: bytes([byte_decoder[token] for token in token_str])
        for token_str, idx in gpt2_vocab.items()
    }

    byte_merges = []
    for m1, m2 in merges:
        b1 = bytes([byte_decoder[t] for t in m1])
        b2 = bytes([byte_decoder[t] for t in m2])
        byte_merges.append((b1, b2))

    return BPETokenizer(vocab, byte_merges, special_tokens=["<|endoftext|>"])


def main():
    if len(sys.argv) < 3:
        print("Usage: python prepare_data.py <input_txt> <output_bin>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    print(f"Loading tokenizer...")
    tokenizer = load_tokenizer()

    print(f"Reading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Tokenizing {len(text)} characters...")
    # This might be slow for huge files, but fine for 5MB
    ids = tokenizer.encode(text)
    print(f"Generated {len(ids)} tokens.")

    # Save as uint16 (GPT-2 vocab is ~50k < 65k)
    print(f"Saving to {output_path}...")
    ids_np = np.array(ids, dtype=np.uint16)
    ids_np.tofile(output_path)
    print("Done.")


if __name__ == "__main__":
    main()
