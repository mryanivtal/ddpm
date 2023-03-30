from pathlib import Path

import torch
from matplotlib import pyplot as plt
import torch.nn.functional as F
from common_utils.torch_utils.torch_pil_utils import display_images_from_tensor
from model_parts.simple_unet import SimpleUnet
from dataloader_utils import get_reverse_image_transforms


def get_loss(noise_scheduler, model, x_0, t, device):
    x_noisy, noise = noise_scheduler.forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    loss = F.l1_loss(noise, noise_pred)
    return loss


@torch.no_grad()
def sample_timestep(noise_scheduler, x, t, model):
    """
    Based on the DDPM paper algorithm 2.
    Uses the model to predict the noise in image x and timestep t, returns denoised image for that timestep.
    if not in the last step, adds noise to the image according to the expected noise level at t-1
    :param x:
    :param t:
    :return:
    """
    betas_t = noise_scheduler.get_betas_t(t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = noise_scheduler.get_sqrt_one_minus_alphas_cumprod_t(t, x.shape)
    sqrt_recip_alphas_t = noise_scheduler.get_sqrt_recip_alphas_t(t, x.shape)
    posterior_variance_t = noise_scheduler.get_posterior_variance_t(t, x.shape)
    # use model - get current image - noise prediction
    model_mean = sqrt_recip_alphas_t * (x - (betas_t / sqrt_one_minus_alphas_cumprod_t) * model(x, t))

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_from_model_and_plot(model, noise_scheduler, timesteps, image_size, device, title=None, display=True, save_path=None, n_columns=8):
    num_images = 5
    image = torch.randn((1, *image_size), device=device)
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    stepsize = int(timesteps / num_images)
    display_tensor = torch.Tensor(0, *image_size).to(device)

    for i in range(0, timesteps)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        image = sample_timestep(noise_scheduler, image, t, model)
        if i % stepsize == 0:
            display_tensor = torch.cat((display_tensor, image), dim=0)

    display_images_from_tensor(display_tensor.detach().cpu(), image_transforms=get_reverse_image_transforms(), title=title, display=display, save_path=save_path, n_columns=n_columns)


def train_batch(data: torch.Tensor, timesteps, model, noise_scheduler, optimizer, device) -> float:
    # ==== (1) Update discriminator network: maximize log(D(x)) + log(1 - D(G(z)))
    # a. Train discriminator with real data
    model.zero_grad()
    data = data.to(device)
    batch_len = len(data)
    t = torch.randint(0, timesteps, (batch_len,), device=device).long()

    batch_loss = get_loss(noise_scheduler, model, data, t, device)
    batch_loss.backward()
    optimizer.step()

    return batch_loss.item()


def create_or_load_model(model_state_dict_path):
    model = SimpleUnet(3, out_dim=1, time_emb_dim=32)
    if model_state_dict_path is not None:
        model_state_dict_path = Path(model_state_dict_path)
        if not model_state_dict_path.exists():
            raise FileNotFoundError(f'mode file {model_state_dict_path} was not found!')

        model.load_state_dict(torch.load(model_state_dict_path))
        model.eval()
    return model


#
#
# def sample_from_generator_and_plot(n_samples, gen_model, device, title=None, path_to_save=None, noise=None):
#     latent_dim = gen_model.latent_dim
#     if noise is not None:
#         assert list(noise.shape) == [n_samples, latent_dim, 1, 1]
#     else:
#         noise = torch.randn([n_samples, latent_dim, 1, 1], device=device)
#
#     sample = gen_model(noise)
#     display_images_from_tensor(sample, title=title, display=False, save_path=path_to_save, n_columns=8)

#
# def weights_init(model):
#     """
#     Initialize model network weights for Conv2d, ConvTranspose2d, BatchNorm modules.
#     :param model:Pytorch model
#     :return: None
#     """
#     classname = model.__class__.__name__
#     if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
#         nn.init.normal_(model.weight.data, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(model.weight.data, 1.0, 0.02)
#         nn.init.constant_(model.bias.data, 0)
