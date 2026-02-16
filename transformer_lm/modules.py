import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LinearModule(nn.Module):
    """Perform a linear transformation on the input data: y = x @ W.T.

    This module applies a linear transformation to the incoming data using a
    weight matrix initialized with a truncated normal distribution.

    Args:
        in_features (int): Size of each input sample (dimension of x).
        out_features (int): Size of each output sample (dimension of y).
        device (torch.device | None): The device on which to store parameters.
        dtype (torch.dtype | None): The data type of the parameters.

    Attributes:
        W (torch.nn.Parameter): The learnable weight matrix of shape
            (out_features, in_features).
    """

    def __init__(
        self,
        in_features,
        out_features,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialize the Linear module and its parameters."""
        super().__init__()
        self.W = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (..., in_features).

        Returns:
            torch.Tensor: Transformed output tensor of shape (..., out_features).
        """
        output = x @ self.W.T
        return output


class EmbeddingModule(nn.Module):
    """
    A simple lookup table that stores embeddings of a fixed dictionary and size.
    This module is often used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding
    word embeddings.
    Args:
        num_embeddings (int): Size of the dictionary of embeddings.
        embedding_dim (int): The size of each embedding vector.
        device (torch.device | None): The device on which to store parameters.
        dtype (torch.dtype | None): The data type of the parameters.
    Attributes:
        weight (torch.nn.Parameter): The learnable weights of the module of shape
            (num_embeddings, embedding_dim).
    """

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        output = self.weight[token_ids]

        return output


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    Args:
        d_model (int): The dimension of the input tensor.
        eps (float): A value added to the denominator for numerical stability.
        device (torch.device | None): The device on which to store parameters.
        dtype (torch.dtype | None): The data type of the parameters.
    Attributes:
        weight (torch.nn.Parameter): The learnable gain parameter of shape (d_model,).
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(self.eps + mean_square)
        output = x / rms * self.weight
        return output


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class RoPE(nn.Module):
    def __init__(self, d_k: int, max_seq_len: int, theta: float = 10000.0, device=None):
        super().__init__()
        theta_i = 1.0 / (
            theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k)
        )

        m = torch.arange(max_seq_len, device=device)

        freqs = torch.outer(m, theta_i)

        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        self.register_buffer("freqs_cis", freqs_cis)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape

        x_complex = torch.view_as_complex(x.float().reshape(*orig_shape[:-1], -1, 2))

        freqs_cis = self.freqs_cis[token_positions]

        if x_complex.shape[-2] != freqs_cis.shape[-2]:
            freqs_cis = freqs_cis.unsqueeze(-2)

        x_rotated = x_complex * freqs_cis

        x_out = torch.view_as_real(x_rotated).flatten(-2)

        return x_out.type_as(x).reshape(orig_shape)


class FeedForward(nn.Module):
    """
    A SwiGLU FeedForward Network.
    Formula: output = w2(SiLU(w1(x)) * w3(x))
    """

    def __init__(self, d_model: int, d_ff: int, device=None):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False, device=device)
        self.w2 = nn.Linear(d_ff, d_model, bias=False, device=device)
        self.w3 = nn.Linear(d_model, d_ff, bias=False, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Scaled Dot Product Attention with Rotary Positional Embeddings (RoPE).

    Args:
        d_model: The dimension of the input and output.
        num_heads: The number of attention heads.
        max_seq_len: The maximum sequence length (used for RoPE cache).
        rope_theta: The theta parameter for RoPE.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
        use_rope: bool = True,
        device=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.use_rope = use_rope

        self.q_proj = nn.Linear(d_model, d_model, bias=False, device=device)
        self.k_proj = nn.Linear(d_model, d_model, bias=False, device=device)
        self.v_proj = nn.Linear(d_model, d_model, bias=False, device=device)
        self.output_proj = nn.Linear(d_model, d_model, bias=False, device=device)

        self.rope = RoPE(self.d_head, max_seq_len, rope_theta, device=device)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        token_positions: torch.Tensor = None,
    ) -> torch.Tensor:
        B, S, D = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(B, S, self.num_heads, self.d_head)
        K = K.view(B, S, self.num_heads, self.d_head)
        V = V.view(B, S, self.num_heads, self.d_head).transpose(1, 2)

        if token_positions is not None and self.use_rope:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)

        # Scaled Dot Product Attention
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -float("inf"))

        attn = F.softmax(scores, dim=-1) @ V

        attn = attn.transpose(1, 2).contiguous().view(B, S, D)
        return self.output_proj(attn)


class TransformerBlock(nn.Module):
    """
    A single Transformer Decoder Block.
    Contains:
    1. RMSNorm -> MultiHeadAttention (with RoPE) -> Residual
    2. RMSNorm -> FeedForward (SwiGLU) -> Residual
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
        use_rmsnorm: bool = True,
        pre_norm: bool = True,
        use_rope: bool = True,
        device=None,
    ):
        super().__init__()
        self.use_rmsnorm = use_rmsnorm
        self.pre_norm = pre_norm
        self.use_rope = use_rope
        if use_rmsnorm:
            self.ln1 = RMSNorm(d_model, eps=1e-5, device=device)
            self.ln2 = RMSNorm(d_model, eps=1e-5, device=device)
        else:
            self.ln1 = nn.Identity()
            self.ln2 = nn.Identity()

        self.attn = MultiHeadAttention(
            d_model,
            num_heads,
            max_seq_len,
            rope_theta,
            use_rope=use_rope,
            device=device,
        )
        self.ffn = FeedForward(d_model, d_ff, device=device)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        token_positions: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.pre_norm:
            x = x + self.attn(self.ln1(x), mask, token_positions)
            x = x + self.ffn(self.ln2(x))
        else:
            x = self.ln1(x + self.attn(x, mask, token_positions))
            x = self.ln2(x + self.ffn(x))
            x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """
    A Transformer Language Model.
    Structure:
    - Embeddings
    - N Transformer Blocks
    - Final RMSNorm
    - Output Head (Unembedding)
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        use_rmsnorm: bool = True,
        pre_norm: bool = True,
        use_rope: bool = True,
        device=None,
    ):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model, device=device)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    context_length,
                    rope_theta,
                    use_rmsnorm=use_rmsnorm,
                    pre_norm=pre_norm,
                    use_rope=use_rope,
                    device=device,
                )
                for _ in range(num_layers)
            ]
        )
        if use_rmsnorm:
            self.ln_final = RMSNorm(d_model, eps=1e-5, device=device)
        else:
            self.ln_final = nn.Identity()
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, device=device)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(token_ids)

        B, S = token_ids.shape
        token_positions = (
            torch.arange(S, device=token_ids.device).unsqueeze(0).expand(B, -1)
        )
        mask = torch.tril(torch.ones(S, S, device=token_ids.device, dtype=torch.bool))

        for layer in self.layers:
            x = layer(x, mask, token_positions)

        x = self.ln_final(x)
        return self.lm_head(x)
