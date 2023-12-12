import torch
import torch.optim as optim
import numpy as np
from loss_functions import vanilla_score_matching
from prior_models import FCScoreNetwork
import os


# Consts
L = 5
lr = 1e-3
dataset_len = 10000
epochs = 200
checkpoint_path = "checkpoints/gaussian.pth"


# Initialize network
score_network = FCScoreNetwork(L)

# Initialize optimizer
optimizer = optim.Adam(score_network.parameters(), lr=lr)

# Initialize data
mu_signal = np.random.randint(0, 5, L ** 2)
cov_signal = np.identity(L ** 2)
data = np.random.multivariate_normal(mu_signal, cov_signal, dataset_len)
dataset = torch.tensor(data).float()

# Training loop
for i_epoch in range(epochs):
    optimizer.zero_grad()
    loss = vanilla_score_matching(score_network, dataset)
    loss.backward()
    optimizer.step()
    if i_epoch % 50 == 0:
        print(loss)

os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
torch.save(score_network.state_dict(), checkpoint_path)
