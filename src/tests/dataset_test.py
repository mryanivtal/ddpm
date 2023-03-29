import unittest
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from src.common_utils.torch_utils.images_in_mem_dataset import ImagesInMemDataset
from src.common_utils.torch_utils.torch_pil_utils import display_images_from_tensor
from src.common_utils.torch_utils.images_dataset import ImagesDataset


class MyTestCase(unittest.TestCase):
    def test_dataset(self):
        DATASET_DIR = '../../../datasets/cats'
        cats_ds = ImagesDataset(DATASET_DIR)

        for i in range(5):
            t = torch.Tensor([i])
            image = cats_ds[t]
            display_images_from_tensor(image)
            plt.show()

        assert len(cats_ds) == 15747


    def test_in_mem_dataset(self):
        DATASET_DIR = '../../../datasets/cats'
        cats_ds = ImagesInMemDataset(DATASET_DIR)

        for i in range(5):
            t = torch.Tensor([i])
            image = cats_ds[t]
            display_images_from_tensor(image)
            plt.show()

        assert len(cats_ds) == 15747