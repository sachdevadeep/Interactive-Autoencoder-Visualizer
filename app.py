
import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import Autoencoder


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Interactive Autoencoder",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Interactive Autoencoder")
st.markdown(
    "Explore image compression, reconstruction "
    "and 2D latent representations."
)

st.divider()


# ==========================================
# 2. DEVICE CONFIGURATION
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

st.sidebar.success(f"Device: {device}")

if torch.cuda.is_available():
    st.sidebar.write(
        torch.cuda.get_device_name(0)
    )


# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():

    model = Autoencoder().to(device)

    model.load_state_dict(
        torch.load(
            "models/autoencoder.pth",
            map_location=device
        )
    )

    model.eval()

    return model


model = load_model()


# ==========================================
# 4. LOAD MNIST DATASET
# ==========================================

@st.cache_data
def load_dataset():

    transform = transforms.ToTensor()

    dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    return dataset


test_dataset = load_dataset()


# ==========================================
# 5. EXTRACT LATENT REPRESENTATIONS
# ==========================================

@st.cache_data
def extract_latent_space():

    loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False
    )

    latent_values = []
    labels = []

    with torch.no_grad():

        for images, targets in loader:

            images = images.to(device)

            latent = model.encode(images)

            latent_values.append(
                latent.cpu().numpy()
            )

            labels.append(
                targets.numpy()
            )

    latent_values = np.concatenate(
        latent_values
    )

    labels = np.concatenate(labels)

    return latent_values, labels


latent, labels = extract_latent_space()


# ==========================================
# 6. DASHBOARD METRICS
# ==========================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Training Images",
    "60,000"
)

col2.metric(
    "Testing Images",
    "10,000"
)

col3.metric(
    "Latent Dimensions",
    "2"
)

col4.metric(
    "Model Parameters",
    f"{sum(p.numel() for p in model.parameters()):,}"
)

st.divider()


# ==========================================
# 7. LATENT SPACE VISUALIZATION
# ==========================================

st.header("📊 2D Latent Space Visualization")

df = pd.DataFrame({

    "Z1": latent[:, 0],

    "Z2": latent[:, 1],

    "Digit": labels.astype(str)

})

fig = px.scatter(

    df,

    x="Z1",

    y="Z2",

    color="Digit",

    title="MNIST Latent Space",

    hover_data=["Digit"],

    opacity=0.65

)

fig.update_traces(
    marker=dict(size=5)
)

fig.update_layout(
    template="plotly_dark",
    height=650
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ==========================================
# 8. GENERATE NEW DIGITS
# ==========================================

st.divider()

st.header("🎨 Generate New Digits")

st.write(
    "Move the latent coordinates to generate "
    "a handwritten digit."
)

col1, col2 = st.columns(2)

with col1:

    z1 = st.slider(

        "Latent Coordinate Z1",

        min_value=float(latent[:, 0].min()),

        max_value=float(latent[:, 0].max()),

        value=float(np.median(latent[:, 0])),

        step=0.1

    )

with col2:

    z2 = st.slider(

        "Latent Coordinate Z2",

        min_value=float(latent[:, 1].min()),

        max_value=float(latent[:, 1].max()),

        value=float(np.median(latent[:, 1])),

        step=0.1

    )


z = torch.tensor(

    [[z1, z2]],

    dtype=torch.float32

).to(device)


with torch.no_grad():

    generated = model.decode(z)


generated = generated.cpu().numpy().reshape(28, 28)


col1, col2 = st.columns([1, 2])

with col1:

    st.image(
        generated,
        caption="Generated Digit",
        width=250,
        clamp=True
    )

with col2:

    st.write("### Latent Coordinates")

    st.write(f"Z1: {z1:.3f}")

    st.write(f"Z2: {z2:.3f}")


# ==========================================
# 9. LATENT SPACE INTERPOLATION
# ==========================================

st.divider()

st.header("🔄 Latent Space Interpolation")

st.write(
    "Explore the transition between two handwritten digits."
)

col1, col2 = st.columns(2)

with col1:

    start_idx = st.number_input(

        "Starting Image Index",

        min_value=0,

        max_value=9999,

        value=0

    )

with col2:

    end_idx = st.number_input(

        "Ending Image Index",

        min_value=0,

        max_value=9999,

        value=100

    )


steps = st.slider(

    "Interpolation Steps",

    min_value=3,

    max_value=15,

    value=10

)


if st.button("Generate Interpolation"):

    image_a, label_a = test_dataset[int(start_idx)]

    image_b, label_b = test_dataset[int(end_idx)]

    image_a = image_a.unsqueeze(0).to(device)

    image_b = image_b.unsqueeze(0).to(device)

    with torch.no_grad():

        z_a = model.encode(image_a)

        z_b = model.encode(image_b)

        alphas = torch.linspace(
            0,
            1,
            steps
        ).to(device)

        interpolated = []

        for alpha in alphas:

            z = (1 - alpha) * z_a + alpha * z_b

            output = model.decode(z)

            interpolated.append(
                output.cpu().squeeze().numpy()
            )


    fig, axes = plt.subplots(
        1,
        steps,
        figsize=(2 * steps, 3)
    )

    for i in range(steps):

        axes[i].imshow(
            interpolated[i],
            cmap="gray"
        )

        axes[i].set_title(
            f"Step {i + 1}"
        )

        axes[i].axis("off")


    plt.suptitle(
        f"Digit {label_a} to Digit {label_b}"
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.success(
        "Interpolation generated successfully!"
    )


# ==========================================
# 10. IMAGE UPLOAD AND RECONSTRUCTION
# ==========================================

st.divider()

st.header("📤 Upload Your Own Digit")

st.write(
    "Upload a handwritten digit image to reconstruct it."
)

uploaded_file = st.file_uploader(

    "Choose an image",

    type=["png", "jpg", "jpeg"]

)


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("L")

    image = image.resize((28, 28))

    image_array = np.array(
        image,
        dtype=np.float32
    )

    # MNIST has white digits on black background.
    # Convert black-background uploads if needed.
    if image_array.mean() > 127:
        image_array = 255 - image_array

    image_array = image_array / 255.0

    image_tensor = torch.tensor(
        image_array,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0).to(device)


    with torch.no_grad():

        reconstructed, uploaded_latent = model(
            image_tensor
        )


    reconstructed = (
        reconstructed.cpu().squeeze().numpy()
    )

    original = image_tensor.cpu().squeeze().numpy()


    col1, col2 = st.columns(2)

    with col1:

        st.image(
            original,
            caption="Uploaded Image",
            width=250
        )

    with col2:

        st.image(
            reconstructed,
            caption="Reconstructed Image",
            width=250
        )


    st.write(
        "Latent Coordinates:",
        uploaded_latent.cpu().numpy()
    )


# ==========================================
# 11. FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    ### About This Project

    Interactive Autoencoder and Latent Space Visualizer

    **Technologies:**
    - Python
    - PyTorch
    - MNIST Dataset
    - NumPy
    - Plotly
    - Streamlit

    Built as a Deep Learning project.
    """
)
