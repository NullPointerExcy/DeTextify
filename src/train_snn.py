import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings

warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    message=r".*PyTree type <class 'norse.*"
)

import torch.nn.functional as F
from PIL import Image, ImageOps
from skimage.metrics import structural_similarity as ssim
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
from safetensors.torch import save_file

from src.utils.image_dataset import ImageDataset
from src.utils.manipulate_train_dataset import add_text_to_image
from src.models.snn_vae_model import SpikingVAE

from src.utils.plot_train_results import plot_train_results_loss, plot_train_results_ssim

TRAINING_SET_DIR = "../dataset/coco"
ORIGINAL_IMAGES_DIR = os.path.join(TRAINING_SET_DIR, "train2017")
MANIPULATED_IMAGES_DIR = os.path.join(TRAINING_SET_DIR, "manipulated_images")

version: str = "0.1.0"
do_manipulate_images: bool = False
batch_size: int = 32
epochs: int = 100
learning_rate: float = 0.02

# For testing purposes, limit the number of images to use
max_images: int | None = None
# How much of the training set to use for validation (0-1)
validation_split: float = 0.2
beta1 = 0.5


def manipulate_images(ends_with: str = '.jpg', text: str = "Random Text"):
    """
    Manipulate the images in the training set by adding text to them
    :param ends_with:
    :param text:
    :return:
    """
    if not os.path.exists(MANIPULATED_IMAGES_DIR):
        os.makedirs(MANIPULATED_IMAGES_DIR)

    image_files = [f for f in os.listdir(ORIGINAL_IMAGES_DIR) if f.endswith(ends_with)]
    with tqdm(total=len(image_files)) as pbar:
        for img_file in image_files:
            img_path = os.path.join(ORIGINAL_IMAGES_DIR, img_file)
            img = Image.open(img_path)
            manipulated_img = add_text_to_image(img.copy(), text=text)
            manipulated_img.save(os.path.join(MANIPULATED_IMAGES_DIR, img_file))
            pbar.update(1)


def pad_to_multiple(img: Image, max_size=(128, 128), multiple=32):
    """
    Pad the image to the nearest multiple of the specified value
    :param img:
    :param max_size: Maximum size of the image
    :param multiple: Multiple to pad to
    :return:
    """
    width, height = img.size

    if width > max_size[0] or height > max_size[1]:
        img.thumbnail(max_size, Image.Resampling.LANCZOS)

    width, height = img.size
    pad_width = (multiple - width % multiple) % multiple
    pad_height = (multiple - height % multiple) % multiple

    padding = (0, 0, pad_width, pad_height)
    return ImageOps.expand(img, padding)


def loss_function(reconstructed_x, x, mu, logvar):
    """
    Calculate the loss function for the SNN-VAE
    Uses binary cross-entropy for the reconstruction loss and KL divergence for the latent space
    :param reconstructed_x:
    :param x:
    :param mu:
    :param logvar:
    :return:
    """
    recon_loss = F.binary_cross_entropy(reconstructed_x, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_divergence


def calculate_ssim(img1, img2):
    """
    Calculate the Structural Similarity Index (SSIM) between two images
    """
    img1 = img1.cpu().numpy().transpose(1, 2, 0)
    img2 = img2.cpu().numpy().transpose(1, 2, 0)

    min_dim = min(img1.shape[0], img1.shape[1])
    win_size = min(7, min_dim)

    return ssim(img1, img2, win_size=win_size, channel_axis=2, data_range=1.0)


def train_snn_vae():
    """
    Train the SpikingVAE model on the image dataset.
    """
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset = ImageDataset(ORIGINAL_IMAGES_DIR, MANIPULATED_IMAGES_DIR, transform=transform)

    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpikingVAE(latent_dim=64, time_steps=50, img_height=128, img_width=128).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    train_losses = []
    val_losses = []
    ssim_scores = []

    print("Starting training...")
    for epoch in range(epochs):
        # ======================================
        # Training phase
        # ======================================
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs, mu, logvar = model(inputs)
            loss = loss_function(outputs, targets, mu, logvar)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))

        # ======================================
        # Validation phase
        # ======================================
        model.eval()
        val_loss = 0.0
        total_ssim = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs, mu, logvar = model(inputs)
                loss = loss_function(outputs, targets, mu, logvar)
                val_loss += loss.item()

                for j in range(inputs.size(0)):
                    total_ssim += calculate_ssim(outputs[j], targets[j])

        scheduler.step(val_loss)

        val_losses.append(val_loss / len(val_loader))
        avg_ssim = total_ssim / len(val_loader.dataset)
        ssim_scores.append(avg_ssim)

        print(
            f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, SSIM: {avg_ssim:.4f}')
        if (epoch + 1) % 10 == 0:
            save_file(model.state_dict(),
                      f"../models/snn_vae_checkpoints/snn_vae_epoch_{epoch + 1}_v{version}.safetensors")

    save_file(model.state_dict(), f"../models/snn_vae_checkpoints/snn_vae_detextify_v{version}.safetensors")

    plot_train_results_loss(train_losses, val_losses)
    plot_train_results_ssim(ssim_scores)

    return model


if __name__ == "__main__":
    # Only manipulate images if the flag is set (to avoid unnecessary processing)
    if do_manipulate_images:
        manipulate_images()
    model = train_snn_vae()
