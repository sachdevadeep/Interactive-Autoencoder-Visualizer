
import torch
import matplotlib.pyplot as plt
import plotly.express as px

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import Autoencoder


# 1. Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# 2. Load MNIST test dataset
transform = transforms.ToTensor()

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False
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


# 4. Extract latent representations
all_latent = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        latent = model.encode(images)

        all_latent.append(latent.cpu())
        all_labels.append(labels)

# Combine all batches
all_latent = torch.cat(all_latent).numpy()
all_labels = torch.cat(all_labels).numpy()

print("Latent shape:", all_latent.shape)
print("Labels shape:", all_labels.shape)


# 5. Create DataFrame
import pandas as pd

df = pd.DataFrame({
    "Z1": all_latent[:, 0],
    "Z2": all_latent[:, 1],
    "Digit": all_labels.astype(str)
})


# 6. Interactive Plotly visualization
fig = px.scatter(
    df,
    x="Z1",
    y="Z2",
    color="Digit",
    title="MNIST - 2D Latent Space Visualization",
    hover_data=["Digit"],
    opacity=0.65
)

fig.update_traces(
    marker=dict(size=5)
)

fig.update_layout(
    width=1000,
    height=700,
    template="plotly_dark"
)

# Save interactive graph
fig.write_html("outputs/latent_space.html")

fig.show()

print("Latent space visualization saved!")
