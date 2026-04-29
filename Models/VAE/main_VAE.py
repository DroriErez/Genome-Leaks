import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models_10K_VAE import *
from pca_plot_genomes import pca_plot
from sklearn.decomposition import PCA
from torch.nn import functional as F

inpt = "Data/1000G_real_genomes/10K_SNP_1000G_real_PADDED_train.hapt" #hdf format input file
eval_inpt = "Data/1000G_real_genomes/10K_SNP_1000G_real_PADDED_eval.hapt" #hdf format input file

out_dir = "./output_dir"
model_name = "VAE_model"
epochs = 10001
lr = 0.001
batch_size = 256
channels = 8
ag_size = 500 #number of artificial genomes (haplotypes) to be created
gpu = 1 #number of GPUs
alph = 0.001 #alpha value for LeakyReLU
save_that = 100 #epoch interval for saving outputs
noise_dim = 1 #dimension of latent_space

device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu > 0) else "cpu")

## Prepare the training data
#df = pd.read_hdf(inpt, key="df1", mode='r')
df = pd.read_csv(inpt, sep = ' ', header=None)
df = df.sample(frac=1).reset_index(drop=True)
df_noname = df.drop(df.columns[0:2], axis=1)
df_noname = df_noname.values
df = df.iloc[0:ag_size,:]
dataloader = torch.utils.data.DataLoader(df_noname, batch_size=batch_size, shuffle=True, pin_memory=True)


df_eval = pd.read_csv(eval_inpt, sep = ' ', header=None)
df_eval = df_eval.sample(frac=1).reset_index(drop=True)
df_eval_noname = df_eval.drop(df_eval.columns[0:2], axis=1)
df_eval_noname = df_eval_noname.values
df_eval = df_eval.iloc[0:ag_size,:]
eval_x = torch.tensor(df_eval_noname, device=device).float()
eval_x = eval_x.reshape(eval_x.shape[0], 1, eval_x.shape[1])

latent_size = int((df_noname.shape[1]+1)/(2**12))

vae = VAE(data_shape=df_noname.shape[1], latent_size=latent_size, channels=channels, noise_dim = noise_dim, alph=alph)
if (device.type == 'cuda') and (gpu > 1):
    vae = nn.DataParallel(vae, list(range(gpu)))
vae.to(device)


# Load model if checkpoint exists
checkpoint_path = f'{out_dir}/{model_name}_last_model'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vae.load_state_dict(checkpoint['VAE'])
    vae.encoder.load_state_dict(checkpoint['Encoder'])
    vae.decoder.load_state_dict(checkpoint['Decoder'])
    start_epoch = checkpoint.get('epoch', 0)
    print(f"Loaded VAE model from {checkpoint_path}, starting from epoch {start_epoch}")
else:
    start_epoch = 0


optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

losses = []

print("Starting Training Loop...")
start_time = time.time()

#Training
for epoch in range(start_epoch, epochs):
    train_loss = 0.0
    n_samples = 0
    for x in dataloader:
        x = x.to(device)
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1])).float()
        optimizer.zero_grad()
        x_hat = vae(x)
        loss = F.binary_cross_entropy(x_hat, x, reduction="sum") + vae.encoder.kl
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        n_samples += x.shape[0]
    train_avg_loss = train_loss / n_samples
    print(f'{model_name}:Epoch: {epoch} \tLoss: {train_loss:.6f}\tAvg Loss: {train_avg_loss:.6f}')
    losses.append(round(train_loss, 3))

    ## Outputs for assessment at every "save_that" epoch
    if epoch%save_that == 0 or epoch == epochs:
        torch.cuda.empty_cache()

        torch.save({
        'epoch': epoch,
        'VAE': vae.state_dict(),
        'Encoder': vae.encoder.state_dict(),
        'Decoder': vae.decoder.state_dict(),
        'data_shape': df_noname.shape[1],
        'latent_size': latent_size,
        'channels': channels,
        'noise_dim': noise_dim,
        'alph': alph},
        f'{out_dir}/{model_name}_{epoch}')

        vae.eval()
        latent_samples = torch.normal(mean=0, std=1, size=(ag_size, 1, latent_size), device = device)
        with torch.no_grad():
            generated_genomes = vae.decoder(latent_samples)
        generated_genomes = generated_genomes.detach().cpu().numpy()
        generated_genomes[generated_genomes < 0] = 0
        generated_genomes = np.rint(generated_genomes)
        generated_genomes = generated_genomes.reshape(generated_genomes.shape[0],generated_genomes.shape[2])
        generated_genomes_df = pd.DataFrame(generated_genomes)
        generated_genomes_df = generated_genomes_df.astype(int)

        gen_names = list()
        for i in range(0,len(generated_genomes_df)):
            gen_names.append('AG'+str(i))
        generated_genomes_df.insert(loc=0, column='Type', value="AG")
        generated_genomes_df.insert(loc=1, column='ID', value=gen_names)
        generated_genomes_df.columns = list(range(generated_genomes_df.shape[1]))
        df.columns = list(range(df.shape[1]))

        generated_genomes_df.to_hdf(f'{out_dir}/{model_name}_{epoch}_output.hapt', key="df1", mode="w")

        pd.DataFrame(losses).to_csv(f'{out_dir}/{model_name}_{epoch}_losses.txt', sep=" ", header=False, index=False)
        fig = plt.figure()
        plt.plot(losses, label='Loss')
        plt.title("Training Loss")
        plt.legend()
        fig.savefig(f'{out_dir}/{model_name}_{epoch}_loss.pdf', format='pdf')
        plt.close(fig)

        # Evaluate in batches to minimize memory usage
        eval_loss = 0.0
        n_eval_samples = 0
        vae.eval()
        batch_eval_size = 128  # Adjust as needed for your GPU/CPU
        with torch.no_grad():
            for i in range(0, eval_x.shape[0], batch_eval_size):
                batch = eval_x[i:i+batch_eval_size]
                eval_x_hat = vae(batch)
                batch_loss = F.binary_cross_entropy(eval_x_hat, batch, reduction="sum") + vae.encoder.kl
                eval_loss += batch_loss.item()
                n_eval_samples += batch.shape[0]

        eval_avg_loss = eval_loss / n_eval_samples

        pca_plot(df, generated_genomes_df, epoch, dir=out_dir, model_name=model_name)

        loss_file = f'{out_dir}/{model_name}_epoch_loss.csv'
        write_header = not os.path.exists(loss_file)
        with open(loss_file, 'a') as f:
            if write_header:
                f.write('epoch,train_loss, eval_loss, train_avg_loss, eval_avg_loss\n')
            f.write(f'{epoch},{train_loss:.6f},{eval_loss:.6f},{train_avg_loss:.6f},{eval_avg_loss:.6f}\n')

        print(f"Epoch: {epoch}, Train Loss: {train_loss:.6f}, Eval Loss: {eval_loss:.6f}, Train Avg Loss: {train_avg_loss:.6f}, Eval Avg Loss: {eval_avg_loss:.6f}")

        torch.save({
        'epoch': epoch,
        'VAE': vae.state_dict(),
        'Encoder': vae.encoder.state_dict(),
        'Decoder': vae.decoder.state_dict(),
        'data_shape': df_noname.shape[1],
        'latent_size': latent_size,
        'channels': channels,
        'noise_dim': noise_dim,
        'alph': alph},
        f'{out_dir}/{model_name}_last_model')


        vae.train()
print("--- %s seconds ---" % (time.time() - start_time))
