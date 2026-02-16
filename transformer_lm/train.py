from tests.adapters import run_save_checkpoint
from torch.nn.functional import cross_entropy
import argparse
import logging
import os
import time
import math

import numpy as np
import torch

from transformer_lm.modules import TransformerLM
from transformer_lm.optimizer import AdamW, get_lr_cosine_schedule, gradient_clipping
from transformer_lm.loggers import ExperimentLogger

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_batch(dataset: np.memmap, batch_size: int, context_length: int, device: str):
    """
    Sample a batch of data from the dataset.

    Args:
        dataset: The memory-mapped numpy array of tokens.
        batch_size: Number of sequences to sample.
        context_length: Length of each sequence.
        device: Device to move the tensors to.

    Returns:
        x: Input tensor of shape (batch, context_length)
        y: Target tensor of shape (batch, context_length)
    """
    # 1. Generate random indices
    # 2. Slice dataset for x and y
    # 3. specific dtype=torch.long
    ix = torch.randint(low=0, high=len(dataset) - context_length, size=(batch_size,))
    x_list = []
    y_list = []
    for i in ix:
        x_list.append(dataset[i : i + context_length])
        y_list.append(dataset[i + 1 : i + context_length + 1])
    x_batch = torch.tensor(np.stack(x_list)).to(dtype=torch.long, device=device)
    y_batch = torch.tensor(np.stack(y_list)).to(dtype=torch.long, device=device)
    return x_batch, y_batch


def train(args):
    """
    Main training loop.
    """
    # 1. Setup Device
    device = args.device
    logger.info(f"Using device: {device}")

    # 2. Load Data
    # Assuming data is stored as uint16 binaries
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    data = np.memmap(args.data_path, dtype=np.uint16, mode="r")
    logger.info(f"Loaded data from {args.data_path}, size: {len(data)} tokens")

    # 3. Model Setup
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        # attn_pdrop=args.attn_pdrop,
        # resid_pdrop=args.resid_pdrop,
        # attn_pdrop=args.attn_pdrop,
        # resid_pdrop=args.resid_pdrop,
        rope_theta=args.rope_theta,
        use_rmsnorm=not args.no_rmsnorm,
        pre_norm=not args.post_norm,
        use_rope=not args.no_rope,
    )
    model.to(device)
    logger.info("Model initialized and moved to device")

    # 4. Optimizer Setup
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # 5. Experiment Logger
    exp_logger = ExperimentLogger(args.out_dir)

    # 6. Compile
    if args.compile:
        logger.info("Compiling model...")
        backend = "aot_eager" if device == "mps" else "inductor"
        try:
            model = torch.compile(model, backend=backend)
            logger.info(f"Model compiled with backend: {backend}")
        except Exception as e:
            logger.warning(f"Compilation failed: {e}")

    # 7. Training Loop
    logger.info("Starting training...")
    model.train()

    from tqdm import tqdm

    pbar = tqdm(range(args.max_iters), desc="Training", dynamic_ncols=True)

    lr = 0.0
    for it in pbar:
        # ... [Existing Training Steps] ...
        # (This block replaces lines 96-97 and sets up for the loop modification below)

        # 1. Get batch
        x, y = get_batch(data, args.batch_size, args.context_length, device)
        # 2. Zero gradients
        if it % args.accumulation_steps == 0:
            optimizer.zero_grad()

        # 3. Forward pass & 4. Calculate loss (Mixed Precision)
        # Determine appropriate dtype for mixed precision
        pt_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

        # Autocast context
        device_type = (
            "cuda"
            if "cuda" in args.device
            else ("cpu" if "cpu" in args.device else "mps")
        )

        with torch.amp.autocast(device_type=device_type, dtype=pt_dtype):
            logits = model(x)
            loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        # Scale loss for gradient accumulation
        loss = loss / args.accumulation_steps

        # 5. Backward pass
        loss.backward()

        # 6. Gradient Clipping, 7. Step, 8. Schedule (only on accumulation boundaries)
        if (it + 1) % args.accumulation_steps == 0:
            gradient_clipping(model.parameters(), args.grad_clip)
            optimizer.step()

            # Update LR
            lr = get_lr_cosine_schedule(
                it, args.lr, args.lr * 0.1, 2000, args.max_iters
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        # Logging
        if it % args.log_interval == 0:
            logger.info(
                f"Iter {it}: Loss {loss.item() * args.accumulation_steps:.4f}, LR {lr:.2e}"
            )
            exp_logger.log_scalar(
                "train_loss", loss.item() * args.accumulation_steps, it
            )
            exp_logger.log_scalar("lr", lr, it)

            pbar.set_description(
                f"Loss: {loss.item() * args.accumulation_steps:.4f} | LR: {lr:.2e}"
            )

        # Validation Loop
        if args.val_data_path and it % args.eval_interval == 0:
            model.eval()
            val_loss = 0.0
            val_steps = 20  # Hardcoded for now, or add arg
            logger.info("Running validation...")
            val_data = np.memmap(args.val_data_path, dtype=np.uint16, mode="r")

            with torch.no_grad():
                for _ in range(val_steps):
                    vx, vy = get_batch(
                        val_data, args.batch_size, args.context_length, device
                    )
                    with torch.amp.autocast(device_type=device_type, dtype=pt_dtype):
                        vlogits = model(vx)
                        vloss = cross_entropy(
                            vlogits.view(-1, vlogits.size(-1)), vy.view(-1)
                        )
                    val_loss += vloss.item()

            avg_val_loss = val_loss / val_steps
            logger.info(f"Iter {it}: Val Loss {avg_val_loss:.4f}")
            exp_logger.log_scalar("val_loss", avg_val_loss, it)
            model.train()

        # Checkpointing
        if it > 0 and it % args.save_interval == 0:
            # ... [Existing Checkpointing] ...
            run_save_checkpoint(
                model, optimizer, it, os.path.join(args.out_dir, f"checkpoint_{it}.pt")
            )

    exp_logger.close()
    logger.info("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a TransformerLM")

    # Data args
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to .bin dataset"
    )
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size")
    parser.add_argument(
        "--context_length", type=int, default=1024, help="Context length"
    )

    # Model args
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--attn_pdrop", type=float, default=0.1)
    parser.add_argument("--resid_pdrop", type=float, default=0.1)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # Optimizer args
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--accumulation_steps", type=int, default=1, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "bfloat16"],
        help="Data type for training",
    )

    # Training args
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "mps"
    )

    # Logging/Saving/Validation
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default="checkpoints")

    parser.add_argument(
        "--val_data_path", type=str, default=None, help="Path to validation data"
    )
    parser.add_argument(
        "--compile", action="store_true", help="Compile model with torch.compile"
    )
    parser.add_argument(
        "--no_rmsnorm", action="store_true", help="Disable RMSNorm (Ablation)"
    )
    parser.add_argument(
        "--post_norm", action="store_true", help="Use Post-Norm architecture (Ablation)"
    )
    parser.add_argument(
        "--no_rope", action="store_true", help="Disable RoPE (Ablation)"
    )

    args = parser.parse_args()

    # Create output dir
    os.makedirs(args.out_dir, exist_ok=True)

    train(args)
