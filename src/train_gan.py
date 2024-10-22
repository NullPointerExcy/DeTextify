import os
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import transforms

from src.models.gan_model import GAN
from src.train_snn import validation_split
from src.utils.GAN.discriminator import Discriminator
from src.utils.image_dataset import ImageDataset
from src.utils.plot_train_results import plot_train_results_gd_loss

TRAINING_SET_DIR = "../dataset/coco"
ORIGINAL_IMAGES_DIR = os.path.join(TRAINING_SET_DIR, "train2017")
MANIPULATED_IMAGES_DIR = os.path.join(TRAINING_SET_DIR, "manipulated_images")


adversarial_loss = nn.BCEWithLogitsLoss()
reconstruction_loss = nn.MSELoss()

# ======================================
# Hyperparameters
# ======================================
lr = 0.0002
beta1 = 0.5
batch_size = 32
num_epochs = 100
generator = GAN().cuda()
discriminator = Discriminator().cuda()


def save_models(generator, discriminator, optimizer_G, optimizer_D, epoch):
    save_dir = "../models/gan_checkpoints/"
    os.makedirs(save_dir, exist_ok=True)

    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
    }, os.path.join(save_dir, f"gan_epoch_{epoch}.pth"))


def load_models(generator, discriminator, optimizer_G, optimizer_D, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    # Load the states of the models and optimizers
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

    epoch = checkpoint['epoch']
    print(f"Loaded model from epoch {epoch}")
    return epoch


if __name__ == '__main__':
    optimizer_G = torch.optim.AdamW(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D = torch.optim.AdamW(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

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

    for epoch in range(num_epochs):
        # Training phase
        generator.train()
        discriminator.train()

        for i, (masked_images, real_images) in enumerate(train_loader):
            masked_images = masked_images.cuda()
            real_images = real_images.cuda()

            # Adversarial ground truths
            valid = torch.ones_like(discriminator(real_images)).cuda()
            fake = torch.zeros_like(discriminator(real_images)).cuda()

            # ======================================
            #  Train Generator
            # ======================================
            optimizer_G.zero_grad()

            # Generate images with text removed
            generated_images = generator(masked_images)

            # Losses
            g_loss_adv = adversarial_loss(discriminator(generated_images), valid)
            g_loss_recon = reconstruction_loss(generated_images, real_images)
            g_loss = g_loss_adv + g_loss_recon

            g_loss.backward()
            optimizer_G.step()

            # ======================================
            #  Train Discriminator
            # ======================================
            optimizer_D.zero_grad()

            # Discriminator loss on real and fake images
            real_loss = adversarial_loss(discriminator(real_images), valid)
            fake_loss = adversarial_loss(discriminator(generated_images.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # Print the losses
            if i % 10 == 0:
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(train_loader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

        # ======================================
        # Validation phase (optional)
        # ======================================
        generator.eval()
        discriminator.eval()

        with torch.no_grad():
            val_g_loss = 0.0
            val_d_loss = 0.0
            for i, (masked_images, real_images) in enumerate(val_loader):
                masked_images = masked_images.cuda()
                real_images = real_images.cuda()

                valid = torch.ones_like(discriminator(real_images)).cuda()
                fake = torch.zeros_like(discriminator(real_images)).cuda()

                # Generate images with text removed
                generated_images = generator(masked_images)

                # Generator validation loss
                g_loss_adv = adversarial_loss(discriminator(generated_images), valid)
                g_loss_recon = reconstruction_loss(generated_images, real_images)
                val_g_loss += (g_loss_adv + g_loss_recon).item()

                # Discriminator validation loss
                real_loss = adversarial_loss(discriminator(real_images), valid)
                fake_loss = adversarial_loss(discriminator(generated_images), fake)
                val_d_loss += (real_loss + fake_loss).item()

            # Calculate average validation loss
            val_g_loss /= len(val_loader)
            val_d_loss /= len(val_loader)
            print(f"[Epoch {epoch}/{num_epochs}] [Validation G loss: {val_g_loss}] [Validation D loss: {val_d_loss}]")

        # Save models
        plot_train_results_gd_loss(val_g_loss, val_d_loss)
        save_models(generator, discriminator, optimizer_G, optimizer_D, epoch)
