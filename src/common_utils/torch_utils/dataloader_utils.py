import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
import torchvision.transforms as transforms
from src.common_utils.torch_utils.images_dataset import ImagesDataset


def seed_init_fn(seed=1):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def create_image_dataloader(dataset_dir: str, batch_size=50, worker_init_fn=seed_init_fn, num_workers=0) -> DataLoader:
    if not Path(dataset_dir).exists():
        raise FileNotFoundError(f'Input data folder does not exist: {Path(dataset_dir).absolute()}')

    data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda t: t/256),         # Scale to [0, 1]
        transforms.Lambda(lambda t: (t * 2) - 1),   # Scale to [-1, 1]
    ])

    image_ds = ImagesDataset(dataset_dir, transforms=data_transforms)
    image_dl = DataLoader(image_ds, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn,
                          num_workers=num_workers)
    return image_dl
