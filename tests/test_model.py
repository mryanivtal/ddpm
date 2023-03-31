import unittest

from src.model_parts.simple_unet import SimpleUnet


class MyTestCase(unittest.TestCase):
    def test_something(self):
        model = SimpleUnet(3, out_dim=1, time_emb_dim=32)
        num_params = sum(p.numel() for p in model.parameters())
        print('Number of params: ', '{:,}'.format(num_params))
        print(model)

        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
