# 1511.06434 UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS
# https://arxiv.org/abs/1511.06434

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from src.ddpm.dataloader_utils import seed_init_fn, create_image_dataloader
from ddpm_functions import train_batch, sample_from_generator_and_plot, weights_init
import warnings

from src.model_parts.dcgan_discriminator import DcganDiscriminator
from src.model_parts.dcgan_generator import DcganGenerator

warnings.simplefilter(action='ignore', category=FutureWarning)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

argparser = argparse.ArgumentParser()
argparser.add_argument('--outdir', type=str, default='./output', help='output folder')
argparser.add_argument('--datadir', type=str, default='../../datasets/cats', help='dataset folder')
argparser.add_argument('--lrgen', type=float, default=2e-4, help='generator learning rate')
argparser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
argparser.add_argument('--batchsize', type=int, default=50, help='train batch size')
argparser.add_argument('--betagen', type=float, default=0.5, help='generator adam beta')
argparser.add_argument('--randomseed', type=int, default=123, help='initial random seed')
argparser.add_argument('--dlworkers', type=int, default=0, help='number of dataloader workers')


args = argparser.parse_args()

OUTPUT_DIR = args.outdir
DATASET_DIR = args.datadir
DIS_LEARNING_RATE = args.lrdis
GEN_LEARNING_RATE = args.lrgen
NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batchsize
DIS_BETA = args.betadis
GEN_BETA = args.betagen
RANDOM_SEED = args.randomseed
DL_WORKERS = args.dlworkers

IMAGE_SIZE = [64, 64]
IMAGE_NUM_CHANNELS = 3
LATENT_DIM = 100
NUM_GEN_FEATURE_MAPS = 64
NUM_DISC_FEATURE_MAPS = 64
REAL_LABEL = 1.
FAKE_LABEL = 0.

# == prep folders ==
output_path = Path(OUTPUT_DIR)
output_path.mkdir(exist_ok=True, parents=True)
print(f'Output path: {output_path.absolute()}')

if RANDOM_SEED is not None:
    seed_init_fn(RANDOM_SEED)
    print(f'Random seed: {RANDOM_SEED}')

print(f'DIS_LEARNING_RATE = {DIS_LEARNING_RATE}')
print(f'GEN_LEARNING_RATE = {GEN_LEARNING_RATE}')
print(f'DIS_BETA = {DIS_BETA}')
print(f'GEN_BETA = {GEN_BETA}')
print(f'NUM_EPOCHS = {NUM_EPOCHS}')
print(f'BATCH_SIZE = {BATCH_SIZE}')
print(f'DL_WORKERS = {DL_WORKERS}')

# == Data ==
cats_dl = create_image_dataloader(DATASET_DIR, batch_size=BATCH_SIZE, num_workers=DL_WORKERS)

# == Model ==
gen_model = DcganGenerator(num_channels=IMAGE_NUM_CHANNELS, num_feature_maps=NUM_GEN_FEATURE_MAPS, latent_dim=LATENT_DIM)
gen_model.apply(weights_init)
gen_model.to(device)

dis_model = DcganDiscriminator(num_channels=IMAGE_NUM_CHANNELS, num_feature_maps=NUM_DISC_FEATURE_MAPS)
dis_model.apply(weights_init)
dis_model.to(device)

# == Optimizer and loss ==
criterion = nn.BCELoss().to(device)
dis_optimizer = torch.optim.Adam(dis_model.parameters(), lr=DIS_LEARNING_RATE, betas=(DIS_BETA, 0.999))
gen_optimizer = torch.optim.Adam(gen_model.parameters(), lr=GEN_LEARNING_RATE, betas=(GEN_BETA, 0.999))

# == Train loop ==
epoch_losses = pd.DataFrame(columns=['epoch', 'gen_loss', 'dis_loss'])
fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=device)
sample_from_generator_and_plot(64, gen_model, device, title=f'Epoch 0', path_to_save=output_path / Path(f'epoch_0'),
                               noise=fixed_noise)

for epoch in tqdm(range(NUM_EPOCHS)):
    epoch += 1
    batch_losses_gen = []
    batch_losses_dis = []

    for i, data in enumerate(cats_dl):
        batch_loss = train_batch(data, gen_model, gen_optimizer, dis_model, dis_optimizer, criterion, device, real_label=REAL_LABEL, fake_label=FAKE_LABEL)
        batch_losses_gen.append(batch_loss['loss_gen'])
        batch_losses_dis.append(batch_loss['loss_dis'])

    epoch_loss = {'epoch': epoch, 'gen_loss': np.average(batch_losses_gen), 'dis_loss': np.average(batch_losses_dis)}
    epoch_losses = epoch_losses.append(epoch_loss, ignore_index=True)
    print(f'Epoch: {epoch}, dis_loss: {epoch_loss["dis_loss"]}, gen_loss: {epoch_loss["gen_loss"]}')
    epoch_losses.to_csv(output_path / Path('train_loss.csv'))
    sample_from_generator_and_plot(64, gen_model, device, title=f'Epoch {epoch}', path_to_save=output_path / Path(f'epoch_{epoch}'), noise=fixed_noise)












