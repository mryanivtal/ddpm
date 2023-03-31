# 1511.06434 UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS
# https://arxiv.org/abs/1511.06434
import time
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from dataloader_utils import seed_init_fn, create_image_dataloader
import warnings

from ddpm_functions import train_batch, sample_from_model_and_plot, \
    save_model_checkpoint, save_optim_checkpoint, \
    update_model_from_checkpoint, update_optim_from_checkpoint
from noise_scheduler import NoiseScheduler
from model_parts.simple_unet import SimpleUnet

warnings.simplefilter(action='ignore', category=FutureWarning)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

argparser = argparse.ArgumentParser()
argparser.add_argument('--outdir', type=str, default='./output', help='output folder')
argparser.add_argument('--datadir', type=str, default='../../datasets/cats', help='dataset folder')
argparser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
argparser.add_argument('--timesteps', type=int, default=300, help='model number of timesteps (T)')
argparser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
argparser.add_argument('--batchsize', type=int, default=50, help='train batch size')
argparser.add_argument('--randomseed', type=int, default=123, help='initial random seed')
argparser.add_argument('--dlworkers', type=int, default=0, help='number of dataloader workers')
argparser.add_argument('--modelcheckpoint', type=str, default=None, help='start from saved model')
argparser.add_argument('--optimcheckpoint', type=str, default=None, help='start from saved optimizer')
argparser.add_argument('--betastart', type=float, default=1e-4, help='diffusion model noise scheduler beta start')
argparser.add_argument('--betaend', type=float, default=2e-2, help='diffusion model noise scheduler beta end')
argparser.add_argument('--checkpointevery', type=int, default=5, help='save checkpoint every N epochs, 0 for disable') #TODO: change to 5 debug
argparser.add_argument('--onebatchperepoch', type=int, default=1, help='For debug purposes')  #TODO: change to 0 debug
argparser.add_argument('--inferonly', type=int, default=0, help='Only sample from model, no training')

args = argparser.parse_args()
ONE_BATCH_PER_EPOCH = args.onebatchperepoch
MODEL_CHECKPOINT = args.modelcheckpoint
OPTIM_CHECKPOINT = args.optimcheckpoint
OUTPUT_DIR = args.outdir
DATASET_DIR = args.datadir
TIMESTEPS = args.timesteps
LEARNING_RATE = args.lr
NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batchsize
RANDOM_SEED = args.randomseed
DL_WORKERS = args.dlworkers
BETA_START = args.betastart
BETA_END = args.betaend
CHECKPOINT_EVERY = args.checkpointevery
INFER_ONLY = args.inferonly

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
print(f'NUM_EPOCHS = {NUM_EPOCHS}')
print(f'BATCH_SIZE = {BATCH_SIZE}')
print(f'DL_WORKERS = {DL_WORKERS}')


# == Data ==
cats_dl = create_image_dataloader(DATASET_DIR, batch_size=BATCH_SIZE, num_workers=DL_WORKERS)

# # todo: delete below ==
# print('Debug mode, limited dataset!!!')
# BATCH_SIZE = 25
# cats_dl.dataset.file_list = cats_dl.dataset.file_list[:50]
# cats_dl.dataset.length = len(cats_dl.dataset.file_list)
# # todo: delete above ==

# == Model and optimizer ==
noise_scheduler = NoiseScheduler(TIMESTEPS, beta_start=BETA_START, beta_end=BETA_END)

model = SimpleUnet(3, out_dim=1, time_emb_dim=32)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

if MODEL_CHECKPOINT is not None:
    update_model_from_checkpoint(model, MODEL_CHECKPOINT, device)

if OPTIM_CHECKPOINT is not None:
    update_optim_from_checkpoint(optimizer, OPTIM_CHECKPOINT, device)

model.to(device)

# == initial sample - before train ==
model.eval()
sample_title = f'Initial sample - Before train'
sample_filename = output_path / Path(f'epoch_0.jpg')
sample_from_model_and_plot(model, noise_scheduler, TIMESTEPS, IMAGE_SIZE, device, title=sample_title,
                           save_path=sample_filename)

if INFER_ONLY == 1:
    time.sleep(20)
    exit(0)

# == Train loop ==
epoch_losses = pd.DataFrame(columns=['epoch', 'loss'])
for epoch in tqdm(range(NUM_EPOCHS)):
    epoch += 1
    batch_losses = []

    model.train()
    for i, data in enumerate(cats_dl):
        batch_loss = train_batch(data, TIMESTEPS, model, noise_scheduler, optimizer, device)
        batch_losses.append(batch_loss)

        if ONE_BATCH_PER_EPOCH:
            break

    # Sample from model, report losses, save checkpoint
    model.eval()
    epoch_loss = np.average(batch_losses)
    epoch_loss_dict = {'epoch': epoch, 'loss': epoch_loss}
    epoch_losses = epoch_losses.append(epoch_loss_dict, ignore_index=True)
    print(f'Epoch: {epoch}, loss: {epoch_loss}')

    losses_filename = output_path / Path('train_loss.csv')
    epoch_losses.to_csv(losses_filename)

    sample_title = f'Epoch {epoch}, loss={epoch_loss}'
    sample_filename = output_path / Path(f'epoch_{epoch}.jpg')
    sample_from_model_and_plot(model, noise_scheduler, TIMESTEPS, IMAGE_SIZE, device, title=sample_title, save_path=sample_filename)

    if CHECKPOINT_EVERY != 0:
        if epoch % CHECKPOINT_EVERY == 0:
            model_path = output_path / Path(f'model_epoch_{epoch}.pt')
            optim_path = output_path / Path(f'optim_epoch_{epoch}.pt')
            save_model_checkpoint(model, model_path)
            save_optim_checkpoint(optimizer, optim_path)



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










