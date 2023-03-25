import unittest
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from src.common_utils.torch_utils.torch_pil_utils import display_images_from_tensor
from src.common_utils.torch_utils.images_dataset import ImagesDataset


class MyTestCase(unittest.TestCase):
    def test_dataset(self):
        cats_ds = ImagesDataset('../../datasets/cats')

        for i in range(5):
            t = torch.Tensor([i])
            image = cats_ds[t]
            display_images_from_tensor(image)
            plt.show()

        assert len(cats_ds) == 15747
