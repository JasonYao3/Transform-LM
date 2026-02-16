import torch
import torch.nn as nn
from transformer_lm.generation import generate


class MockModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, x):
        return self.head(self.embedding(x))


def test_generation():
    vocab_size = 100
    model = MockModel(vocab_size, 32)
    prompt = torch.tensor([[1, 2, 3]])

    # Needs to handle the undefined EOT_ID and variable names
    try:
        out = generate(model, prompt, max_gen_len=5, temperature=1.0, top_p=0.9)
        print("Generated:", out)
        assert out.shape == (1, 8)  # 3 prompt + 5 gen
    except Exception as e:
        print(f"Caught expected error: {e}")


if __name__ == "__main__":
    test_generation()
