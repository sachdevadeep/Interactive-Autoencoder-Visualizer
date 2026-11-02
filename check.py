
import torch
from model import Autoencoder

model = Autoencoder()

# Create a dummy batch of 128 images
x = torch.randn(128, 1, 28, 28)

# Forward pass
reconstructed, latent = model(x)

print("Input shape:", x.shape)
print("Latent shape:", latent.shape)
print("Output shape:", reconstructed.shape)

print("Total parameters:",
      sum(p.numel() for p in model.parameters()))
