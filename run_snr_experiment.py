import argparse
import sys
import torchvision
import numpy as np
import utils
import approximate_EM
import matplotlib.pyplot as plt
from prior_models import *


def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset to be used')
    parser.add_argument('--checkpoint_path', type=str, default="checkpoints/mnist.pth", help='Prior model weights')
    parser.add_argument('--trials', type=int, default=10,  help='Numbers of target images to average')
    parser.add_argument('--random_seed', type=int, default=1337, help='Random seed')
    parser.add_argument('--epsilon', type=float, default=1e-5, help='Stopping criteria')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--K', type=int, default=4, help='Number of rotations to consider')
    parser.add_argument('--em_iterations', type=int, default=100, help='Number of EM iterations')
    parser.add_argument('--snrs', type=list, default=[0.1, 0.5, 1], help='SNRs to examine')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    np.random.seed(args.random_seed)

    if args.dataset == "MNIST":
        # Prepare data
        L = 28
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        mnist_dataset = torchvision.datasets.MNIST("mnist", download=True, transform=transforms)

        # Prepare initializator
        alpha_initial = np.sum([np.array(mnist_dataset[i][0][0]) for i in np.random.randint(len(mnist_dataset), size=10)], axis=0) / 10

        # Build dataset
        dataset = [np.array(mnist_dataset[i][0][0]) for i in np.random.randint(len(mnist_dataset), size=args.trials)]  # Be sure to exclude from prior model training

        # Prepare gammas
        gammas = np.linspace(0, 1, args.em_iterations)

        # Load prior model class
        score_network = UnetScoreNetwork()
    else:
        # Prepare data
        L = 5
        mu_signal = np.random.randint(0, 5, L ** 2)  # Be sure to align with prior model training
        cov_signal = np.identity(L ** 2)

        # Prepare initializator
        alpha_initial = np.random.rand(L, L)

        # Build dataset
        dataset = np.random.multivariate_normal(mu_signal, cov_signal, args.trials).reshape(args.trials, L, L)

        # Prepare gammas
        gammas = np.ones(args.em_iterations)  # Multiply by 10 for enhanced performance

        # Load prior model class
        score_network = FCScoreNetwork()

    # Load prior model weights
    state_dict = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
    score_network.load_state_dict(state_dict)
    score_network.eval()

    errs_with_prior = []
    errs_without_prior = []
    for snr in args.snrs:
        errs_with_prior_per_snr = []
        errs_without_prior_per_snr = []
        for i in range(args.trials):
            X = dataset[i]
            sigma_noise = np.sqrt(np.linalg.norm(X) ** 2 / (snr * L ** 2))
            M = utils.create_measurement(X, sigma_noise, args.K)

            # With prior
            alpha_with_prior = approximate_EM.EM(M, alpha_initial, sigma_noise, gammas, args.K, args.em_iterations, args.epsilon, args.lr, prior=score_network)
            err_with_prior = utils.calc_error(X, alpha_with_prior, args.K)
            errs_with_prior_per_snr.append(err_with_prior)

            # Without prior
            alpha_without_prior = approximate_EM.EM(M, alpha_initial, sigma_noise, gammas, args.K, args.em_iterations, args.epsilon, args.lr, prior=None)
            err_without_prior = utils.calc_error(X, alpha_without_prior, args.K)
            errs_without_prior_per_snr.append(err_without_prior)

        errs_with_prior.append(np.mean(errs_with_prior_per_snr))
        errs_without_prior.append(np.mean(errs_without_prior_per_snr))

    plt.figure()
    plt.plot(args.snrs, errs_with_prior, color='r', label="Approximate EM with score-based prior")
    plt.scatter(args.snrs, errs_with_prior, color='r')
    plt.plot(args.snrs, errs_without_prior, color='b', label="Approximate EM")
    plt.scatter(args.snrs, errs_without_prior, color='b')

    plt.xlabel("SNR")
    plt.ylabel("Mean estimation error")
    plt.xticks(args.snrs)
    plt.legend(fontsize=7)
    plt.show()

    return 0


if __name__ == '__main__':
    sys.exit(main())
