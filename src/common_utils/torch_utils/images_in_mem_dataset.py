import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from pathlib import Path


class ImagesInMemDataset(Dataset):
    def __init__(self, images_dir: str, transforms=None):
        self.images_path = Path(images_dir)
        self.transforms = transforms
        assert self.images_path.exists()
        self.data = []

        file_list = list(self.images_path.glob('*'))
        for image_path in file_list:
            sample = read_image(str(image_path))
            if self.transforms:
                sample = self.transforms(sample)

            self.data += sample[None, :]

        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = int(item.item())

        return self.data[item]



