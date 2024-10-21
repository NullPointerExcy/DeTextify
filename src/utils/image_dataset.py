import os
from torch.utils.data import Dataset
from PIL import Image


class ImageDataset(Dataset):

    def __init__(self, original_dir, manipulated_dir, transform=None, max_images=1000, ends_with='.jpg'):
        self.original_dir = original_dir
        self.manipulated_dir = manipulated_dir
        self.image_files = [f for f in os.listdir(original_dir) if f.endswith(ends_with)]

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