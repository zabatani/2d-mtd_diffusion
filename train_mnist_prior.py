import torch
import torchvision
from prior_models import UnetScoreNetwork
from loss_functions import denoising_score_matching
import os


# Consts
lr = 3e-4
batch_size = 64
epochs = 400
checkpoint_path = "checkpoints/mnist.pth"


# Initialize network
score_network = UnetScoreNetwork()

# Initialize optimizer
optimizer = torch.optim.Adam(score_network.parameters(), lr=lr)

# Initialize data
transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])
data = torchvision.datasets.MNIST("mnist", download=True, transform=transforms)
data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

# Check for CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
score_network = score_network.to(device)

# Training loop
for i_epoch in range(epochs):
    total_loss = 0
    for batch, _ in data_loader:
        batch = batch.reshape(batch.shape[0], -1).to(device)
        optimizer.zero_grad()
        loss = denoising_score_matching(score_network, batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * batch.shape[0]
    if i_epoch % 50 == 0:
        print(total_loss)

os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
torch.save(score_network.state_dict(), checkpoint_path)
