import os
import sys
import numpy as np
import tiktoken


def main():
    if len(sys.argv) < 3:
        print("Usage: python prepare_data_fast.py <input_txt> <output_prefix>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_prefix = sys.argv[2]

    print("Loading tiktoken 'gpt2' encoding...")
    enc = tiktoken.get_encoding("gpt2")

    print(f"Reading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Tokenizing {len(text)} characters...")
    # allowed_special={'<|endoftext|>'} is crucial if the text contains it
    ids = enc.encode(text, allowed_special={"<|endoftext|>"})
    print(f"Generated {len(ids)} tokens.")

    # Split 90/10
    n = len(ids)
    train_ids = ids[: int(n * 0.9)]
    val_ids = ids[int(n * 0.9) :]

    print(f"Train split: {len(train_ids)} tokens")
    print(f"Val split: {len(val_ids)} tokens")

    # Save train
    train_path = f"{output_prefix}_train.bin"""
    print(f"Saving to {train_path}...")
    train_ids_np = np.array(train_ids, dtype=np.uint16)
    train_ids_np.tofile(train_path)

    # Save val
    val_path = f"{output_prefix}_val.bin"
    print(f"Saving to {val_path}...")
    val_ids_np = np.array(val_ids, dtype=np.uint16)
    val_ids_np.tofile(val_path)

    print("Done.")


if __name__ == "__main__":
    main()
