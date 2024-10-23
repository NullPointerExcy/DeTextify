import os
import sys
import torch
from PIL import Image
from torchvision import transforms
from datetime import datetime
from src.models.gan_model import GAN
from src.utils.GAN.discriminator import Discriminator
from src.train_gan import load_models


MODELS_DIR = "models/gan_checkpoints"


def load_latest_gan_model(generator, discriminator, optimizer_G, optimizer_D, models_dir):
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]

    if not model_files:
        raise FileNotFoundError("No model found in the models directory.")

    # Sort by last modified time, descending (most recent first)
    model_files.sort(key=lambda f: os.path.getmtime(os.path.join(models_dir, f)), reverse=True)

    latest_model_path = os.path.join(models_dir, model_files[0])
    print(f"Loading latest model: {latest_model_path}")

    # Load the models and optimizers
    epoch = load_models(generator, discriminator, optimizer_G, optimizer_D, latest_model_path)

    generator.eval()
    return generator


def process_image(image_path, generator):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image {image_path} not found.")

    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = generator.to(device)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        generated_image = generator(input_tensor)

    # Convert tensor back to image and save
    output_image = transforms.ToPILImage()(generated_image.squeeze(0).cpu())  # Remove batch dimension

    output_image_path = f"outputs/output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    output_image.save(output_image_path)

    print(f"Processed image saved as: {output_image_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_gan.py <input_image_path> [<model_version>, default=latest]")
        sys.exit(1)

    input_image_path = sys.argv[1]

    generator = GAN()
    discriminator = Discriminator()
    optimizer_G = torch.optim.AdamW(generator.parameters())
    optimizer_D = torch.optim.AdamW(discriminator.parameters())

    try:
        # Load the latest GAN model
        generator = load_latest_gan_model(generator, discriminator, optimizer_G, optimizer_D, MODELS_DIR)
        process_image(input_image_path, generator)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
