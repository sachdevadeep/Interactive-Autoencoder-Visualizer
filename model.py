
import torch
import torch.nn as nn


class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder: Compress image
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),

            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, 2)
        )

        # Decoder: Reconstruct image
        self.decoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),

            nn.Linear(64, 256),
            nn.ReLU(),

            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    # Convert image into latent vector
    def encode(self, x):
        x = x.view(x.size(0), 784)
        return self.encoder(x)

    # Convert latent vector into image
    def decode(self, z):
        return self.decoder(z)

    # Complete forward pass
    def forward(self, x):
        z = self.encode(x)
        reconstructed = self.decode(z)

        reconstructed = reconstructed.view(-1, 1, 28, 28)

        return reconstructed, z
