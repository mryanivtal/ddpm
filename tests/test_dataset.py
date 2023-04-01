import unittest
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import transforms

from src.utils.torch_utils.images_dataloader_utils import create_image_dataloader, get_image_transforms, \
    get_reverse_image_transforms
from src.utils.torch_utils.images_dataset import ImagesDataset
from src.utils.torch_utils.images_in_mem_dataset import ImagesInMemDataset
from src.utils.torch_utils.torch_pil_utils import display_images_from_tensor


class DatasetTestCase(unittest.TestCase):
    def test_dataset(self):
        DATASET_DIR = '../../datasets/cats'
        cats_ds = ImagesDataset(DATASET_DIR)

        for i in range(5):
            t = torch.Tensor([i])
            image = cats_ds[t]
            display_images_from_tensor(image)
            plt.show()

        assert len(cats_ds) == 15747


    def test_in_mem_dataset(self):
        DATASET_DIR = '../../datasets/cats'
        cats_ds = ImagesInMemDataset(DATASET_DIR)

        for i in range(5):
            t = torch.Tensor([i])
            image = cats_ds[t]
            display_images_from_tensor(image)
            plt.show()

        assert len(cats_ds) == 15747


    def test_dataloader(self):
        DATASET_DIR = '../../datasets/cats'
        BATCH_SIZE = 5

        cats_dl = create_image_dataloader(DATASET_DIR, batch_size=BATCH_SIZE)
        images = cats_dl.__iter__().__next__()

        inverse_transforms = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t * 256),
            transforms.Lambda(lambda t: t.type(torch.uint8))
        ])

        for i in range(BATCH_SIZE):
            image = images[i]
            display_images_from_tensor(image, image_transforms=inverse_transforms)
            plt.show()

        assert (True)


    def test_transforms(self):
        DATASET_DIR = '../../datasets/cats'
        cats_ds = ImagesInMemDataset(DATASET_DIR)

        data_transforms = get_image_transforms()
        inverse_transforms = get_reverse_image_transforms()

        image = cats_ds[0]
        image_b = inverse_transforms(data_transforms(image))

        display_images_from_tensor(image)
        display_images_from_tensor(image_b)
        display_images_from_tensor(data_transforms(image), image_transforms=inverse_transforms)

        diff = image_b - image
        assert(diff.sum().item() == 0)
