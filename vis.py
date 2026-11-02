
import torch
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import Autoencoder


# 1. Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 2. Load test dataset
transform = transforms.ToTensor()

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=10,
    shuffle=True
)

# 3. Load trained model
model = Autoencoder().to(device)

model.load_state_dict(
    torch.load(
        "models/autoencoder.pth",
        map_location=device
    )
)

model.eval()

# 4. Get test images
images, labels = next(iter(test_loader))

images = images.to(device)

# 5. Reconstruction
with torch.no_grad():
    reconstructed, latent = model(images)

# 6. Visualize
fig, axes = plt.subplots(2, 10, figsize=(15, 4))

for i in range(10):

    # Original image
    axes[0, i].imshow(
        images[i].cpu().squeeze(),
        cmap="gray"
    )

    axes[0, i].set_title(f"Original: {labels[i].item()}")
    axes[0, i].axis("off")

    # Reconstructed image
    axes[1, i].imshow(
        reconstructed[i].cpu().squeeze(),
        cmap="gray"
    )

    axes[1, i].set_title("Reconstructed")
    axes[1, i].axis("off")

plt.suptitle("Original vs Reconstructed Images")

plt.tight_layout()

plt.savefig("outputs/reconstruction.png")

plt.show()

print("Reconstruction completed!")
print("Latent shape:", latent.shape)
