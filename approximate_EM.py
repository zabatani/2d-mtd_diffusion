import torch
import numpy as np
import itertools


def patch_generator(M, L):
    N = M.shape[0]
    patches_in_axes = N // L
    for i in range(0, patches_in_axes):
        for j in range(0, patches_in_axes):
            patch = np.array([M[row_idx][j * L:(j + 1) * L] for row_idx in range(i * L, (i + 1) * L)])
            yield patch


def CTZ(F, k, l, L, zero_pad=True):
    if zero_pad:
        ZF = np.zeros((2 * L, 2 * L))
    else:
        ZF = np.empty((2 * L, 2 * L))
        ZF[:] = np.nan
    ZF[:L, :L] = np.rot90(F, k)
    TZF = np.roll(ZF, l, axis=(0, 1))
    CTZF = TZF[:L, :L]
    return CTZF


def EM(M_noised, alpha, sigma_noise, gammas, K, em_iterations, epsilon, lr, prior):
    L = alpha.shape[0]
    ls = list(itertools.product(np.arange(2 * L), np.arange(2 * L)))
    alpha = np.copy(alpha)
    for i in range(em_iterations):
        grad = np.zeros((L, L))

        # TODO: vectorize this section
        masks = np.zeros((len(ls), K, L, L))
        new_ind = np.empty((len(ls), L), dtype=object)
        where = np.empty((len(ls), L), dtype=object)
        for l in range(len(ls)):
            for k in range(K):
                masks[l, k] = CTZ(alpha, k, ls[l], L)
                transformation = CTZ(np.array(range(L ** 2)).reshape(L, L), k, ls[l], L, zero_pad=False)
                new_ind[l, k] = np.unravel_index(transformation[~np.isnan(transformation)].astype(int), (L, L))
                where[l, k] = np.where(~np.isnan(transformation))

        for patch in patch_generator(M_noised, L):
            # Calcuate weights
            log_likelihoods = -np.sum((patch - masks) ** 2, axis=(2, 3)) / (2 * sigma_noise ** 2)
            weights = np.exp(log_likelihoods, dtype=np.float128)
            weights = weights / np.sum(weights)

            # Build gradients
            for l in range(len(ls)):
                for k in range(K):
                    grad[new_ind[l, k]] += weights[l, k] * (
                                (patch[where[l, k]] - alpha[new_ind[l, k]]) / sigma_noise ** 2)

        if prior:
            prior_grad = prior(torch.from_numpy(alpha.flatten()).unsqueeze(0).float()).detach().numpy().reshape(L, L)
            grad += gammas[i] * prior_grad

        # Gradient ascent (single) step
        alpha += lr * grad

        if lr * np.linalg.norm(grad) < epsilon:
            break

    return alpha
