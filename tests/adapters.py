from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor
import math
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule


import regex as re
from collections import defaultdict

from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.modules import (
    LinearModule,
    EmbeddingModule,
    RMSNorm,
    SiLU,
    RoPE,
    TransformerLM,
)


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    lm = LinearModule(d_in, d_out)
    lm.load_state_dict({"W": weights})
    return lm(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    em = EmbeddingModule(vocab_size, d_model)
    em.load_state_dict({"weight": weights})

    return em(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight

    # We can use the FeedForward class we implemented in modules.py
    # But FeedForward was not imported? Let's check imports.
    # It is imported as part of `cs336_basics.modules` but likely not exposed as `FeedForward`
    # if I didn't add it to __all__ or import it explicitly?
    # Actually I can just use nn.Linear and F.silu directly here for simplicity
    # since this is just an adapter test for the logic.
    # OR better, instantiate FeedForward from modules.

    # Let's check if FeedForward is imported.
    # In Step 1124 I imported LinearModule, EmbeddingModule, RMSNorm, SiLU, RoPE, TransformerLM.
    # I did NOT import FeedForward!

    # I will implement it manually here to avoid import issues,
    # or I can update imports. Updating imports is cleaner.
    # But for now, let's just implement the math: output = w2(SiLU(w1(x)) * w3(x))

    # Check shapes:
    # w1: (d_ff, d_model) -> (d_model, d_ff) for linear
    # w2: (d_model, d_ff) -> (d_ff, d_model) for linear
    # w3: (d_ff, d_model) -> (d_model, d_ff) for linear

    # F.linear uses (x @ weight.T)
    # w1_weight is (d_ff, d_model). So x @ w1_weight.T is (..., d_ff). Correct.

    a1 = F.linear(in_features, w1_weight)
    a3 = F.linear(in_features, w3_weight)
    z = F.silu(a1) * a3
    return F.linear(z, w2_weight)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    S = Q @ K.transpose(-2, -1)
    d_k = Q.size(-1)
    S = S / math.sqrt(d_k)
    if mask is not None:
        S = S.masked_fill(mask == 0, -float("inf"))

    P = torch.softmax(S, dim=-1)
    output = P @ V

    return output


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    Q = F.linear(in_features, q_proj_weight)
    K = F.linear(in_features, k_proj_weight)
    V = F.linear(in_features, v_proj_weight)

    d_head = d_model // num_heads

    Q = Q.view(*Q.shape[:-1], num_heads, d_head).transpose(1, 2)
    K = K.view(*K.shape[:-1], num_heads, d_head).transpose(1, 2)
    V = V.view(*V.shape[:-1], num_heads, d_head).transpose(1, 2)

    seq_len = in_features.shape[-2]
    mask = torch.tril(
        torch.ones(seq_len, seq_len, device=in_features.device), diagonal=0
    ).bool()

    attn_out = run_scaled_dot_product_attention(Q, K, V, mask)

    attn_out = attn_out.transpose(1, 2).contiguous()

    attn_out = attn_out.view(*attn_out.shape[:-2], d_model)
    return F.linear(attn_out, o_proj_weight)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    Q = F.linear(in_features, q_proj_weight)
    K = F.linear(in_features, k_proj_weight)
    V = F.linear(in_features, v_proj_weight)

    d_head = d_model // num_heads

    # 1. Reshape
    Q = Q.view(*Q.shape[:-1], num_heads, d_head).transpose(1, 2)
    K = K.view(*K.shape[:-1], num_heads, d_head).transpose(1, 2)
    V = V.view(*V.shape[:-1], num_heads, d_head).transpose(1, 2)

    # 2. Apply RoPE (The new part!)
    if token_positions is not None:
        Q = run_rope(d_head, theta, max_seq_len, Q, token_positions)
        K = run_rope(d_head, theta, max_seq_len, K, token_positions)

    # 3. Attention
    seq_len = in_features.shape[-2]
    mask = torch.tril(
        torch.ones(seq_len, seq_len, device=in_features.device), diagonal=0
    ).bool()

    attn_out = run_scaled_dot_product_attention(Q, K, V, mask)

    # 4. Output
    attn_out = attn_out.transpose(1, 2).contiguous()
    attn_out = attn_out.view(*attn_out.shape[:-2], d_model)

    return F.linear(attn_out, o_proj_weight)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    rope_module = RoPE(d_k, max_seq_len, theta)

    return rope_module(in_query_or_key, token_positions)


def run_rms_norm(
    x: Float[Tensor, " ... d_model"],
    weight: Float[Tensor, " d_model"],
    eps: float,
) -> Float[Tensor, " ... d_model"]:
    model = RMSNorm(x.shape[-1], eps)
    model.weight.data = weight
    return model(x)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    norm_x = run_rms_norm(in_features, weights["ln1.weight"], 1e-5)

    seq_len = in_features.shape[1]
    token_positions = (
        torch.arange(seq_len, device=in_features.device)
        .unsqueeze(0)
        .expand(in_features.shape[0], -1)
    )

    attn_out = run_multihead_self_attention_with_rope(
        d_model,
        num_heads,
        max_seq_len,
        theta,
        weights["attn.q_proj.weight"],
        weights["attn.k_proj.weight"],
        weights["attn.v_proj.weight"],
        weights["attn.output_proj.weight"],
        norm_x,
        token_positions,
    )

    h = in_features + attn_out

    norm_h = run_rms_norm(h, weights["ln2.weight"], 1e-5)

    a1 = F.linear(norm_h, weights["ffn.w1.weight"])
    a3 = F.linear(norm_h, weights["ffn.w3.weight"])

    ff_inner = F.silu(a1) * a3
    ff_out = F.linear(ff_inner, weights["ffn.w2.weight"])

    out = h + ff_out

    return out


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    model = TransformerLM(
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        rope_theta,
    )
    model.load_state_dict(weights, strict=False)
    return model(in_indices)


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    rmsnorm = RMSNorm(d_model, eps)
    rmsnorm.load_state_dict({"weight": weights})

    return rmsnorm(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    return SiLU()(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    ix = torch.randint(low=0, high=len(dataset) - context_length, size=(batch_size,))
    x_list = []
    y_list = []
    for i in ix:
        x_list.append(dataset[i : i + context_length])
        y_list.append(dataset[i + 1 : i + context_length + 1])
    x_batch = torch.tensor(np.stack(x_list)).to(dtype=torch.long, device=device)
    y_batch = torch.tensor(np.stack(y_list)).to(dtype=torch.long, device=device)
    return x_batch, y_batch


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    return torch.nn.functional.softmax(in_features, dim=dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    return F.cross_entropy(inputs, targets)


def run_gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float
) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    raise NotImplementedError


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """

    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    return get_lr_cosine_schedule(
        it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters
    )


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter": iteration,
    }
    torch.save(checkpoint, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iter"]


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """

    return BPETokenizer(vocab, merges, special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    # vocabulary initialization
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])

    # pre-tokenization
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # process text in input_path
    with open(input_path, encoding="utf-8") as f:
        text = f.read()
        tokens = re.findall(PAT, text)

    # Convert each token to a list of bytes, preserving chunk boundaries
    words = []
    for token in tokens:
        token_bytes = token.encode("utf-8")
        # Convert byte sequence to list of individual bytes
        word = [bytes([b]) for b in token_bytes]
        words.append(word)

    merge_list = []

    # Calculate how many merges we need (accounting for special tokens that will be added)
    num_special_to_add = 0
    special_token_bytes_list = []
    for special_token in special_tokens:
        special_bytes = special_token.encode("utf-8")
        if special_bytes not in vocab.values():
            num_special_to_add += 1
        special_token_bytes_list.append(special_bytes)

    target_vocab_size = vocab_size - num_special_to_add

    # Check if special tokens appear in training text (to know if we need to protect them)
    text_bytes = text.encode("utf-8")
    special_tokens_in_text = [st for st in special_token_bytes_list if st in text_bytes]
    # Pre-compute forbidden substrings (2+ bytes) from special tokens for efficient checking
    forbidden_substrings = set()
    for st_bytes in special_tokens_in_text:
        # Add all substrings of length 2 or more (but not the full token)
        for i in range(len(st_bytes) - 1):
            for j in range(i + 2, len(st_bytes)):
                forbidden_substrings.add(st_bytes[i:j])

    # BPE merging loop
    while len(vocab) < target_vocab_size:
        # Count adjacent pairs within each chunk
        pair_counts = defaultdict(int)
        for word in words:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += 1

        # Stop if no pairs found
        if not pair_counts:
            break

        # Find the most frequent pair
        if special_tokens_in_text:
            # Skip pairs that would create tokens containing parts of special tokens
            sorted_pairs = sorted(
                pair_counts.items(),
                key=lambda x: (
                    -x[1],
                    x[0][0],
                    x[0][1],
                ),  # Negative count for descending
            )

            most_freq_pair = None
            merged_token = None
            for pair, count in sorted_pairs:
                candidate = pair[0] + pair[1]
                # Skip if candidate contains any forbidden substring OR is a substring of special token
                is_forbidden = any(f in candidate for f in forbidden_substrings)
                if not is_forbidden:
                    # Also check if candidate is a proper substring of any special token
                    for st_bytes in special_tokens_in_text:
                        if candidate in st_bytes and candidate != st_bytes:
                            is_forbidden = True
                            break
                if not is_forbidden:
                    most_freq_pair = pair
                    merged_token = candidate
                    break

            # If no valid pair found, stop
            if most_freq_pair is None:
                break
        else:
            # No special tokens in text, use simple max approach
            most_freq_pair = max(
                pair_counts.items(), key=lambda x: (x[1], x[0][0], x[0][1])
            )[0]
            merged_token = most_freq_pair[0] + most_freq_pair[1]

        # Add merged token to vocabulary
        next_id = len(vocab)
        vocab[next_id] = merged_token

        # Record the merge
        merge_list.append(most_freq_pair)

        # Replace all occurrences of the pair with the merged token in all words
        pair_first = most_freq_pair[0]
        pair_second = most_freq_pair[1]
        for word_idx, word in enumerate(words):
            if len(word) < 2:
                continue
            # Build new word with replacements in a single pass
            new_word = []
            i = 0
            changed = False
            while i < len(word):
                if (
                    i < len(word) - 1
                    and word[i] == pair_first
                    and word[i + 1] == pair_second
                ):
                    new_word.append(merged_token)
                    i += 2
                    changed = True
                else:
                    new_word.append(word[i])
                    i += 1
            # Only update if word actually changed (optimization)
            if changed:
                words[word_idx] = new_word

    # Add special tokens to vocabulary
    for special_token in special_tokens:
        special_bytes = special_token.encode("utf-8")
        # Check if special token already exists in vocab
        if special_bytes not in vocab.values():
            next_id = len(vocab)
            vocab[next_id] = special_bytes

    return vocab, merge_list
