import unittest

import torch

from src.utils.torch_utils.images_dataloader_utils import get_reverse_image_transforms, create_image_dataloader
from src.utils.torch_utils.torch_pil_utils import display_images_from_tensor
from src.noise_scheduler import NoiseScheduler


class NoiseSchedulerTestCase(unittest.TestCase):

    def test_noise_scheduler(self):
        DATASET_DIR = '../../datasets/cats'
        BATCH_SIZE = 5
        T_STEPS = 300

        cats_dl = create_image_dataloader(DATASET_DIR, batch_size=BATCH_SIZE)
        image = cats_dl.__iter__().__next__()[0]
        repeats = [1] * (len(image.shape) + 1)
        repeats[0] = T_STEPS
        images = image.repeat(repeats)

        noise_scheduler = NoiseScheduler(T_STEPS)
        noise_steps = torch.arange(T_STEPS)
        noisy_images, noise = noise_scheduler.forward_diffusion_sample(images, noise_steps)

        inverse_transforms = get_reverse_image_transforms()
        display_images_from_tensor(images, image_transforms=inverse_transforms, n_columns=5)
        display_images_from_tensor(noisy_images, image_transforms=inverse_transforms,n_columns=5)

        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
