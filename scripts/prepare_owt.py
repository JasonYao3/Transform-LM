import os
import tiktoken
import numpy as np
from tqdm import tqdm


def prepare_split(split, max_tokens=None):
    input_file = f"data/owt_{split}.txt"
    output_file = f"owt_{split}.bin"

    if not os.path.exists(input_file):
        print(f"Skipping {split}: {input_file} not found")
        return

    print(f"Processing {input_file} -> {output_file}...")
    enc = tiktoken.get_encoding("gpt2")

    # We will write to a temp file first or just build in memory?
    # 50M tokens * 2 bytes = 100MB. Memory is fine.

    all_tokens = []
    token_count = 0

    with open(input_file, "r", encoding="utf-8") as f:
        pbar = tqdm(total=max_tokens if max_tokens else os.path.getsize(input_file))

        for line in f:
            tokens = enc.encode_ordinary(line)
            all_tokens.extend(tokens)
            token_count += len(tokens)
            pbar.update(len(line))  # rough progress

            if max_tokens and token_count >= max_tokens:
                print(f"Reached max tokens limit: {max_tokens}")
                break
        pbar.close()

    # Convert to uint16
    print(f"Saving {token_count} tokens to {output_file}...")
    token_ids = np.array(all_tokens, dtype=np.uint16)

    # Verify range
    if (token_ids > 65535).any():
        print(
            "WARNING: Vocabulary size exceeds uint16 limit! (GPT-2 is 50257, so this should be fine)"
        )

    # Write to file
    fp = np.memmap(output_file, dtype=np.uint16, mode="w+", shape=(len(token_ids),))
    fp[:] = token_ids[:]
    fp.flush()
    print(f"Done. Saved {output_file}")


if __name__ == "__main__":
    # Prepare Train (limit 50M tokens -> ~100MB)
    prepare_split("train", max_tokens=50_000_000)

    # Prepare Valid (limit 5M tokens -> ~10MB)
    prepare_split("valid", max_tokens=5_000_000)
