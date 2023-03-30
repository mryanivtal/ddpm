# 1511.06434 UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS
# https://arxiv.org/abs/1511.06434

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from .dataloader_utils import seed_init_fn, create_image_dataloader
import warnings

from .ddpm_functions import get_loss, train_batch, sample_from_model_and_plot
from .noise_scheduler import NoiseScheduler
from .model_parts.simple_unet import SimpleUnet

warnings.simplefilter(action='ignore', category=FutureWarning)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

argparser = argparse.ArgumentParser()
argparser.add_argument('--outdir', type=str, default='./output', help='output folder')
argparser.add_argument('--datadir', type=str, default='../../datasets/cats', help='dataset folder')
argparser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
argparser.add_argument('--timesteps', type=int, default=100, help='model number of timesteps (T)')
argparser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
argparser.add_argument('--batchsize', type=int, default=50, help='train batch size')
argparser.add_argument('--beta', type=float, default=0.5, help='adam beta')
argparser.add_argument('--randomseed', type=int, default=123, help='initial random seed')
argparser.add_argument('--dlworkers', type=int, default=0, help='number of dataloader workers')


args = argparser.parse_args()

OUTPUT_DIR = args.outdir
DATASET_DIR = args.datadir
TIMESTEPS = args.timesteps
LEARNING_RATE = args.lr
NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batchsize
BETA = args.beta
RANDOM_SEED = args.randomseed
DL_WORKERS = args.dlworkers

IMAGE_NUM_CHANNELS = 3
IMAGE_SIZE = [IMAGE_NUM_CHANNELS, 64, 64]
LATENT_DIM = 100


# == prep folders ==
output_path = Path(OUTPUT_DIR)
output_path.mkdir(exist_ok=True, parents=True)
print(f'Output path: {output_path.absolute()}')

if RANDOM_SEED is not None:
    seed_init_fn(RANDOM_SEED)
    print(f'Random seed: {RANDOM_SEED}')

print(f'GEN_LEARNING_RATE = {LEARNING_RATE}')
print(f'GEN_BETA = {BETA}')
print(f'NUM_EPOCHS = {NUM_EPOCHS}')
print(f'BATCH_SIZE = {BATCH_SIZE}')
print(f'DL_WORKERS = {DL_WORKERS}')

# == Data ==
cats_dl = create_image_dataloader(DATASET_DIR, batch_size=BATCH_SIZE, num_workers=DL_WORKERS)

# == Model ==
model = SimpleUnet(3, out_dim=1, time_emb_dim=32)
noise_scheduler = NoiseScheduler(TIMESTEPS)
model.to(device)

# == Optimizer and loss ==
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(BETA, 0.999))

# == Train loop ==
epoch_losses = pd.DataFrame(columns=['epoch', 'loss'])
for epoch in tqdm(range(NUM_EPOCHS)):
    epoch += 1
    batch_losses = []

    for i, data in enumerate(cats_dl):
        batch_loss = train_batch(data, TIMESTEPS, model, noise_scheduler, optimizer, device)
        batch_losses.append(batch_loss)

        # todo: remove!
        break

    epoch_loss = {'epoch': epoch, 'loss': np.average(batch_losses)}
    epoch_losses = epoch_losses.append(epoch_loss, ignore_index=True)
    print(f'Epoch: {epoch}, loss: {epoch_loss}')


    model_filename = output_path / Path(f'model_epoch_{epoch}.pt')
    torch.save(model.state_dict(), model_filename)

    losses_filename = output_path / Path('train_loss.csv')
    epoch_losses.to_csv(losses_filename)

    sample_title = f'Epoch {epoch}, loss={epoch_loss}'
    sample_filename = output_path / Path(f'epoch_{epoch}.jpg')
    sample_from_model_and_plot(model, noise_scheduler, TIMESTEPS, IMAGE_SIZE, device, title=sample_title, save_path=sample_filename)




# sample_from_generator_and_plot(64, model, device, title=f'Epoch 0', path_to_save=output_path / Path(f'epoch_0'),
#                                noise=fixed_noise)
#
# for epoch in tqdm(range(NUM_EPOCHS)):
#     epoch += 1
#     batch_losses_gen = []
#     batch_losses_dis = []
#
#     for i, data in enumerate(cats_dl):
#         batch_loss = train_batch(data, model, gen_optimizer, dis_model, dis_optimizer, criterion, device, real_label=REAL_LABEL, fake_label=FAKE_LABEL)
#         batch_losses_gen.append(batch_loss['loss_gen'])
#         batch_losses_dis.append(batch_loss['loss_dis'])
#
#     epoch_loss = {'epoch': epoch, 'gen_loss': np.average(batch_losses_gen), 'dis_loss': np.average(batch_losses_dis)}
#     epoch_losses = epoch_losses.append(epoch_loss, ignore_index=True)
#     print(f'Epoch: {epoch}, dis_loss: {epoch_loss["dis_loss"]}, gen_loss: {epoch_loss["gen_loss"]}')
#     epoch_losses.to_csv(output_path / Path('train_loss.csv'))
#     sample_from_generator_and_plot(64, model, device, title=f'Epoch {epoch}', path_to_save=output_path / Path(f'epoch_{epoch}'), noise=fixed_noise)
#
#










