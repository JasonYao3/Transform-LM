from sympy.codegen.ast import none
import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: iter,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for g in group["params"]:
                if g.grad is None:
                    continue

                state = self.state[g]
                # Check if this is the first time we've seen this param
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(g)
                    state["exp_avg_sq"] = torch.zeros_like(g)

                state["step"] += 1
                beta1, beta2 = group["betas"]

                g.data -= group["lr"] * group["weight_decay"] * g.data
                state["exp_avg"] = beta1 * state["exp_avg"] + (1 - beta1) * g.grad
                state["exp_avg_sq"] = (
                    beta2 * state["exp_avg_sq"] + (1 - beta2) * g.grad**2
                )

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                m_hat = state["exp_avg"] / bias_correction1
                v_hat = state["exp_avg_sq"] / bias_correction2
                denom = v_hat.sqrt() + group["eps"]

                g.data -= group["lr"] * m_hat / denom


def get_lr_cosine_schedule(
    it: int,
    max_lr: float,
    min_lr: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return max_lr * (it / warmup_iters)

    if it < cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cos_factor = math.cos(math.pi * progress)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + cos_factor)
    else:
        return min_lr


def gradient_clipping(params: iter, max_norm: float):
    params = list(params)

    grads = [p.grad for p in params if p.grad is not None]

    total_norm_sq = sum(g.norm(2) ** 2 for g in grads)
    total_norm = total_norm_sq.sqrt()

    clip_coef = max_norm / (total_norm + 1e-6)

    if clip_coef < 1.0:
        for g in grads:
            g.data.mul_(clip_coef)
