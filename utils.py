import numpy as np


def calc_error(X, alpha, K):
    errs = []
    for k in range(K):
        err = np.linalg.norm(np.rot90(alpha, k) - X) / np.linalg.norm(X)
        errs.append(err)
    return np.min(errs)


def create_measurement(X, sigma_noise, K, size_magnifier=11, density=0.1):
    L = np.shape(X)[0]
    N = size_magnifier * L
    images_to_plant = round(density * (N / L) ** 2)
    bf_max_iterations = images_to_plant * 10
    successful_plants = []
    M = np.zeros([N, N])

    # Brute-force planting
    for _ in range(bf_max_iterations):
        if len(successful_plants) > images_to_plant:
            break
        candidate = np.random.randint(N - L + 1, size=2)
        if any([abs(plant[0] - candidate[0]) < 2 * L and abs(plant[1] - candidate[1]) < 2 * L for plant in
                successful_plants]):
            continue
        successful_plants.append(candidate)
        M[candidate[0]: candidate[0] + L, candidate[1]: candidate[1] + L] = np.rot90(X, np.random.randint(K))

    M_noised = M + np.random.default_rng().normal(loc=0, scale=sigma_noise, size=np.shape(M))
    return M_noised
