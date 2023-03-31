import unittest

import torch
from matplotlib import pyplot as plt

from src.model_parts.unet_parts import SinusoidalPositionEmbedding

class TimeEmbeddingTestCase(unittest.TestCase):
    def test_time_embedding(self):
        timesteps = 300
        time_emb_dim = 100

        sin_pos_embedding = SinusoidalPositionEmbedding(time_emb_dim)
        time = torch.arange(timesteps)
        embeddings = sin_pos_embedding(time)
        embeddings_left_to_rigth = embeddings.transpose(1, 0)

        fig, ax = plt.subplots(figsize=(timesteps/10, time_emb_dim/10))
        ax.imshow(embeddings_left_to_rigth)
        plt.show()




