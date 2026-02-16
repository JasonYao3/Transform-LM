import torch
import json
import os
from cs336_basics.modules import TransformerLM
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.generation import generate
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode
import argparse


def load_tokenizer():
    vocab_path = FIXTURES_PATH / "gpt2_vocab.json"
    merges_path = FIXTURES_PATH / "gpt2_merges.txt"

    # 1. Load raw vocab (unicode strings -> IDs)
    with open(vocab_path) as f:
        gpt2_vocab = json.load(f)

    # 2. Load merges
    merges = []
    with open(merges_path) as f:
        for line in f:
            line = line.strip()
            if line and len(line.split()) == 2:
                merges.append(tuple(line.split()))

    # 3. Remap vocab to bytes (reversing the GPT-2 specific byte-to-unicode mapping)
    byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    vocab = {
        idx: bytes([byte_decoder[token] for token in token_str])
        for token_str, idx in gpt2_vocab.items()
    }

    # 4. Remap merges to bytes
    byte_merges = []
    for m1, m2 in merges:
        b1 = bytes([byte_decoder[t] for t in m1])
        b2 = bytes([byte_decoder[t] for t in m2])
        byte_merges.append((b1, b2))

    return BPETokenizer(vocab, byte_merges, special_tokens=["<|endoftext|>"])


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained model")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--prompt", type=str, default="Once upon a time", help="Input prompt"
    )
    parser.add_argument("--max_len", type=int, default=256, help="Max output tokens")
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "mps"
    )
    parser.add_argument("--no_rmsnorm", action="store_true", help="Disable RMSNorm")
    args = parser.parse_args()

    device = args.device
    print(f"Using device: {device}")

    # 1. Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer()

    # 2. Initialize Model (Architecture must match training!)
    # TODO: Load config from checkpoint if possible, for now hardcoding to match train.py defaults
    print("Initializing model...")
    model = TransformerLM(
        vocab_size=50257,
        context_length=1024,  # Note: train.py used 256 for sweeping, but Let's stick to 1024 or make it arg
        d_model=512,  # train.py sweep used 512
        num_layers=8,  # train.py sweep used 8
        num_heads=8,  # train.py sweep used 8
        d_ff=2048,  # 4 * d_model
        rope_theta=10000.0,
        use_rmsnorm=not args.no_rmsnorm,
        device=device,
    )

    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=device)
        # Handle state dict key mismatch if needed (e.g. if saved with 'module.')
        state_dict = ckpt["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model.eval()

    # 3. Encode Prompt
    print(f"Prompt: {args.prompt}")
    print("Encoding...")
    input_ids = tokenizer.encode(args.prompt)
    x = torch.tensor([input_ids], dtype=torch.long, device=device)

    # 4. Generate
    print("Generating...")
    out_ids = generate(
        model,
        x,
        max_gen_len=args.max_len,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # 5. Decode
    print("Decoding...")
    # Convert tensor back to list of ints
    out_list = out_ids[0].tolist()
    generated_text = tokenizer.decode(out_list)

    print(f"\nGenerated Text:\n{generated_text}")


if __name__ == "__main__":
    main()
