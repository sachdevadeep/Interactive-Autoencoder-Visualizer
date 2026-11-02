
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import Autoencoder

import os
import time


# --------------------------------
# 1. Device configuration
# --------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Training Device:", device)


# --------------------------------
# 2. Dataset loading
# --------------------------------

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)


# --------------------------------
# 3. Initialize model
# --------------------------------

model = Autoencoder().to(device)

# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# --------------------------------
# 4. Training configuration
# --------------------------------

epochs = 10

train_losses = []
test_losses = []

start_time = time.time()


# --------------------------------
# 5. Training loop
# --------------------------------

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)

        # Forward pass
        reconstructed, latent = model(images)

        # Calculate reconstruction loss
        loss = criterion(reconstructed, images)

        # Reset gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    train_losses.append(avg_train_loss)


    # --------------------------------
    # 6. Testing / Evaluation
    # --------------------------------

    model.eval()

    test_running_loss = 0.0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            reconstructed, latent = model(images)

            loss = criterion(reconstructed, images)

            test_running_loss += loss.item()

    avg_test_loss = test_running_loss / len(test_loader)

    test_losses.append(avg_test_loss)


    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Train Loss: {avg_train_loss:.6f} "
        f"Test Loss: {avg_test_loss:.6f}"
    )


# --------------------------------
# 7. Save trained model
# --------------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

torch.save(
    model.state_dict(),
    "models/autoencoder.pth"
)

print("\nModel saved successfully!")


# --------------------------------
# 8. Plot training graph
# --------------------------------

plt.figure(figsize=(9, 5))

plt.plot(train_losses, label="Training Loss")
plt.plot(test_losses, label="Testing Loss")

plt.xlabel("Epochs")
plt.ylabel("MSE Loss")

plt.title("Autoencoder Training Performance")

plt.legend()
plt.grid(True)

plt.savefig("outputs/loss_curve.png")

plt.show()


print("\nTotal Training Time:",
      round(time.time() - start_time, 2), "seconds")
