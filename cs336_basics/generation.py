import torch
import torch.nn.functional as F


def generate(model, prompt, max_gen_len, temperature=1.0, top_p=0.9):
    """
    Generate text from a model.

    Args:
        model: The Transformer model.
        prompt: A tensor of shape (1, seq_len) containing the prompt tokens.
        max_gen_len: Maximum number of tokens to generate.
        temperature: Temperature for sampling.
        top_p: Top-p (nucleus) sampling threshold.

    Returns:
        torch.Tensor: The generated sequence (1, seq_len + gen_len).
    """
    # Clone the prompt so we don't modify the original
    x = prompt.clone()
    EOT_ID = 50256
    for _ in range(max_gen_len):
        logits = model(x)[:, -1, :]

        logits = logits / temperature

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        x = torch.cat([x, next_token], dim=-1)

        if next_token.item() == EOT_ID:
            break

    return x
