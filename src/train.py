import os
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.manipulator.manipulate_train_dataset import add_text_to_image
from src.model import UNetVAE


TRAINING_SET_DIR = "../dataset/coco"
ORIGINAL_IMAGES_DIR = os.path.join(TRAINING_SET_DIR, "train2017")
MANIPULATED_IMAGES_DIR = os.path.join(TRAINING_SET_DIR, "manipulated_images")

version: str = "0.1.0"
do_manipulate_images: bool = False
batch_size: int = 64
epochs: int = 100
max_images: int | None = 4000


def manipulate_images():
    if not os.path.exists(MANIPULATED_IMAGES_DIR):
        os.makedirs(MANIPULATED_IMAGES_DIR)

    image_files = [f for f in os.listdir(ORIGINAL_IMAGES_DIR) if f.endswith('.jpg')]
    with tqdm(total=len(image_files)) as pbar:
        for img_file in image_files:
            img_path = os.path.join(ORIGINAL_IMAGES_DIR, img_file)
            img = Image.open(img_path)
            manipulated_img = add_text_to_image(img.copy(), text="Random Text")
            manipulated_img.save(os.path.join(MANIPULATED_IMAGES_DIR, img_file))
            pbar.update(1)


class ImageDataset(Dataset):
    def __init__(self, original_dir, manipulated_dir, transform=None):
        self.original_dir = original_dir
        self.manipulated_dir = manipulated_dir
        self.image_files = [f for f in os.listdir(original_dir) if f.endswith('.jpg')]

        if max_images is not None:
            self.image_files = self.image_files[:max_images]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        original_img_path = os.path.join(self.original_dir, img_file)
        manipulated_img_path = os.path.join(self.manipulated_dir, img_file)

        original_img = Image.open(original_img_path).convert('RGB')
        manipulated_img = Image.open(manipulated_img_path).convert('RGB')

        if self.transform:
            original_img = self.transform(original_img)
            manipulated_img = self.transform(manipulated_img)

        return manipulated_img, original_img


def loss_function(reconstructed_x, x, mu, logvar):
    recon_loss = F.mse_loss(reconstructed_x, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_divergence


def pad_to_multiple(img, multiple=32):
    width, height = img.size
    pad_width = (multiple - width % multiple) % multiple
    pad_height = (multiple - height % multiple) % multiple

    padding = (0, 0, pad_width, pad_height)
    return F.pad(img, padding, mode='constant', value=0)


def train_unet():
    transform = transforms.Compose([
        transforms.Lambda(lambda img: pad_to_multiple(img, multiple=32)),
        transforms.ToTensor(),
    ])
    dataset = ImageDataset(ORIGINAL_IMAGES_DIR, MANIPULATED_IMAGES_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetVAE().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs, mu, logvar = model(inputs)
            loss = loss_function(outputs, targets, mu, logvar)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}')

    torch.save(model.state_dict(), f"unet_text_removal_v{version}.pth")
    return model


if __name__ == "__main__":
    if do_manipulate_images:
        manipulate_images()
    model = train_unet()
