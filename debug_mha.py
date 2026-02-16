import torch
import torch.nn.functional as F
from tests.adapters import run_multihead_self_attention


def test_manual():
    B, S, D = 1, 12, 64
    H = 4
    in_features = torch.randn(B, S, D)
    q_w = torch.randn(D, D)
    k_w = torch.randn(D, D)
    v_w = torch.randn(D, D)
    o_w = torch.randn(D, D)

    print(f"Testing MHA with B={B}, S={S}, D={D}, H={H}")

    try:
        out = run_multihead_self_attention(
            d_model=D,
            num_heads=H,
            q_proj_weight=q_w,
            k_proj_weight=k_w,
            v_proj_weight=v_w,
            o_proj_weight=o_w,
            in_features=in_features,
        )
        print(f"Output shape: {out.shape}")
    except Exception as e:
        print(f"Failed: {e}")


if __name__ == "__main__":
    test_manual()
