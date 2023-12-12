import torch.autograd as autograd
import torch


def calc_jacobian(f, x):
    B, N = x.shape
    y = f(x)
    jacobian = []
    for i in range(N):
        v = torch.zeros_like(y)
        v[:, i] = 1.
        dy_i_dx = autograd.grad(y, x, grad_outputs=v, retain_graph=True, create_graph=True, allow_unused=True)[0]
        jacobian.append(dy_i_dx)
    jacobian = torch.stack(jacobian, dim=2).requires_grad_()
    return jacobian


def vanilla_score_matching(score_network, x):
    x.requires_grad_(True)
    score = score_network(x)

    # Compute the norm term
    norm = torch.norm(score, dim=-1) ** 2

    # Compute the Jacobian term
    jacobian = calc_jacobian(score_network, x)
    tr_jacobian = torch.diagonal(jacobian, dim1=-2, dim2=-1).sum(-1)

    # Compute loss
    loss = 0.5 * norm + tr_jacobian
    return torch.mean(loss)


def denoising_score_matching(score_network, x, eps=1e-4):
    # Sample time
    t = torch.rand((x.shape[0], 1), dtype=x.dtype, device=x.device) * (1 - eps) + eps

    # Compute log posterior terms
    int_beta = (0.1 + 0.5 * (20 - 0.1) * t) * t
    mu_t = x * torch.exp(-0.5 * int_beta)
    var_t = -torch.expm1(-int_beta)
    x_t = torch.randn_like(x) * var_t ** 0.5 + mu_t
    grad_log_p = -(x_t - mu_t) / var_t

    # Compute loss
    score = score_network(x_t, t)
    loss = (score - grad_log_p) ** 2
    lambda_t = var_t
    weighted_loss = lambda_t * loss
    return torch.mean(weighted_loss)