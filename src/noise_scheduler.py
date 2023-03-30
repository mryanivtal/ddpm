import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from src.common_utils.torch_utils.torch_pil_utils import display_images_from_tensor


class NoiseScheduler:
    def __init__(self, timesteps: int, beta_start=1e-4, beta_end=2e-2):
        self.timesteps = timesteps
        self.betas = self._linear_beta_Schedule(timesteps, start=beta_start, end=beta_end)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(self.alphas_cumprod)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def _linear_beta_Schedule(self, timesteps: int, start=1e-4, end=2e-2) -> torch.Tensor:
        """
        Retuns a linspace tensor based on the arguments for beta_t values
        :param timesteps: No. of timesteps to return (capital T)
        :param start: beta_1
        :param end: beta_T
        :return: tensor of ordered beta_t values
        """
        return torch.linspace(start, end, timesteps)

    def _get_index_from_list(self, vals, t, x_shape):
        """
        returns a specific index t of a passed list of values vals
        while considering the batch dimensions
        :param vals:
        :param t:
        :param x_shape:
        :return:
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.type(torch.int64).cpu())
        return out.reshape(batch_size, *((1,) *  (len(x_shape) - 1))).to(t.device)

    def forward_diffusion_sample(self, x_0, t, device='cpu'):
        """
        Takes an image and a timestamp as input, returns a noisy version of it according to the equation:
        q(x_t | x_0) = N(x_t; sqrt(alphas_cumprod_t) * x_0 , (1 - alphas_cumprod_t) * I )
        :param x_0: original image from dataset
        :param t: timestep for noising x_t
        :param device:
        :return: x_t
        """
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self._get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


    def get_betas_t(self, t, x_shape):
        return self._get_index_from_list(self.betas, t, x_shape)

    def get_sqrt_one_minus_alphas_cumprod_t(self, t, x_shape):
        return self._get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_shape)

    def get_sqrt_recip_alphas_t(self, t, x_shape):
        return self._get_index_from_list(self.sqrt_recip_alphas, t, x_shape)

    def get_posterior_variance_t(self, t, x_shape):
        return self._get_index_from_list(self.posterior_variance, t, x_shape)










