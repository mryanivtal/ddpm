import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from pathlib import Path


class ImagesDataset(Dataset):
    def __init__(self, images_dir: str, transforms=None):
        self.images_path = Path(images_dir)
        self.transforms = transforms
        assert self.images_path.exists()

        self.file_list = list(self.images_path.glob('*'))
        self.length = len(self.file_list)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = int(item.item())

        image_path = Path(self.file_list[item])
        sample = read_image(str(image_path))
        if self.transforms:
            sample = self.transforms(sample)

        return sample


