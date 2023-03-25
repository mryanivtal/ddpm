import unittest

import torch
from model_parts.discriminator_model import Discriminator
from model_parts.generator_model import Generator
from common_utils import common_logger

logger = common_logger.get_logger(__name__)

class TestModels(unittest.TestCase):

    def test_model_shapes(self):
        IMAGE_SIZE = [64, 64]
        NUM_CHANNELS = 3
        BATCH_SIZE = 10

        LATENT_DIM = 100
        NUM_GEN_FEATURE_MAPS = 64
        NUM_DISC_FEATURE_MAPS = 64

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        gen = Generator(num_channels=NUM_CHANNELS, num_feature_maps=NUM_GEN_FEATURE_MAPS, latent_dim=LATENT_DIM)
        dis = Discriminator(num_channels=NUM_CHANNELS, num_feature_maps=NUM_DISC_FEATURE_MAPS)

        gen.to(device)
        dis.to(device)

        print('---------------------------------------------------------------')
        print(dis)
        print('---------------------------------------------------------------')

        img = torch.rand([BATCH_SIZE, NUM_CHANNELS, *IMAGE_SIZE])
        z = torch.rand(BATCH_SIZE, LATENT_DIM, 1, 1)

        print(z.shape)
        print(img.shape)
        print(gen(z).shape)
        print(dis(img).shape)
        print(dis(gen(z)).shape)

        assert list(dis(img).shape) == [BATCH_SIZE, 1, 1, 1]
        assert gen(z).shape == img.shape
        assert list(gen(z).shape) == [BATCH_SIZE, NUM_CHANNELS, *IMAGE_SIZE]
