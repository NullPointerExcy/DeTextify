import os
import sys
import torch
from PIL import Image
from torchvision import transforms
from datetime import datetime
from safetensors import safe_open
from src.models.snn_vae_model import SNNVAE
from src.train import pad_to_multiple

MODELS_DIR = "models"


def load_latest_model(models_dir):
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.safetensors')]

    if not model_files:
        raise FileNotFoundError("No model found in the models directory.")

    model_files.sort(key=lambda f: os.path.getmtime(os.path.join(models_dir, f)), reverse=True)

    latest_model_path = os.path.join(models_dir, model_files[0])
    print(f"Loading latest model: {latest_model_path}")

    model = SNNVAE()
    with safe_open(latest_model_path, framework="pt", device="cpu") as f:
        state_dict = {key: f.get_tensor(key) for key in f.keys()}
        model.load_state_dict(state_dict)

    model.eval()
    return model


def process_image(image_path, model):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image {image_path} not found.")

    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Lambda(lambda img: pad_to_multiple(img, multiple=32)),
        transforms.ToTensor(),
    ])

    input_tensor = transform(image).unsqueeze(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output_tensor, _, _ = model(input_tensor)

    output_image = transforms.ToPILImage()(output_tensor.squeeze(0).cpu())

    output_image_path = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    output_image.save(output_image_path)

    print(f"Processed image saved as: {output_image_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: py run.py <input_image_path> [<model_version>, default=latest]")
        sys.exit(1)

    input_image_path = sys.argv[1]

    try:
        model = load_latest_model(MODELS_DIR)
        process_image(input_image_path, model)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
