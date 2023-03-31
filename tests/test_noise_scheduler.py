import unittest

import torch
from torchvision.transforms import transforms

from src.common_utils.torch_utils.torch_pil_utils import display_images_from_tensor
from src.dataloader_utils import create_image_dataloader
from src.noise_scheduler import NoiseScheduler


class MyTestCase(unittest.TestCase):

    def test_noise_scheduler(self):
        DATASET_DIR = '../../datasets/cats'
        BATCH_SIZE = 5
        T_STEPS = 80

        cats_dl = create_image_dataloader(DATASET_DIR, batch_size=BATCH_SIZE)
        image = cats_dl.__iter__().__next__()[0]
        repeats = [1] * (len(image.shape) + 1)
        repeats[0] = T_STEPS
        images = image.repeat(repeats)

        noise_scheduler = NoiseScheduler(T_STEPS, beta_start=1e-4, beta_end=5e-2)
        noise_steps = torch.arange(T_STEPS)
        noisy_images = noise_scheduler.forward_diffusion_sample(images, noise_steps)

        inverse_transforms = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t * 256),
            transforms.Lambda(lambda t: t.type(torch.uint8))
        ])

        display_images_from_tensor(images, image_transforms=inverse_transforms, n_columns=5)
        display_images_from_tensor(noisy_images, image_transforms=inverse_transforms,n_columns=5)

        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
