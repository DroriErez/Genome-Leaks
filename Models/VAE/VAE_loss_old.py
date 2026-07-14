"""
Evaluate VAE reconstruction losses across saved training checkpoints.

For each synthetic output file in ``out_dir``, this script:
1. Extracts the epoch number from the synthetic file name.
2. Loads the matching VAE checkpoint for that epoch.
3. Computes binary cross-entropy reconstruction losses for train, test, and
   synthetic genomes in small batches.
4. Uses the synthetic average loss as a membership threshold and reports
   train/test membership statistics to ``VAE_loss_over_epochs.csv``.
"""

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
from torch.nn import functional as F
import re
import glob

# Input genotype files. The first two columns are metadata and are removed
# before converting the SNP values to model tensors.
train_inpt = "Data/1000G_real_genomes/10K_SNP_1000G_real_PADDED_train.hapt"
test_inpt = "Data/1000G_real_genomes/10K_SNP_1000G_real_PADDED_test.hapt"

# Directory containing saved checkpoints and generated synthetic HDF outputs.
out_dir = "./output_dir"

loss_file = f"{out_dir}/VAE_loss_over_epochs.csv"
model_name = "VAE_model"
epochs = 10001
lr = 0.001
batch_size = 8 #number of samples per forward pass during loss evaluation
channels = 8 #number of convolution channels used by the VAE
gpu = 1 #number of GPUs to use; set to 0 to force CPU
alph = 0.001 #LeakyReLU negative slope used by the VAE
save_that = 100 #epoch interval used when training saved model outputs
noise_dim = 1 #latent noise dimension expected by the VAE

write_header = True

device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu > 0) else "cpu")

# Load real train data and reshape to (samples, channels, SNPs), matching the
# input shape expected by the convolutional VAE.
df_train = pd.read_csv(train_inpt, sep=' ', header=None)
df_train = df_train.sample(frac=1).reset_index(drop=True)
df_train_noname = df_train.drop(df_train.columns[0:2], axis=1)
df_train_noname = df_train_noname.values
train_x = torch.tensor(df_train_noname, device=device).float()
train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1])

# Load real holdout/test data with the same preprocessing as the train split.
df_test = pd.read_csv(test_inpt, sep = ' ', header=None)
df_test = df_test.sample(frac=1).reset_index(drop=True)
df_test_noname = df_test.drop(df_test.columns[0:2], axis=1)
df_test_noname = df_test_noname.values
test_x = torch.tensor(df_test_noname, device=device).float()
test_x = test_x.reshape(test_x.shape[0], 1, test_x.shape[1])

# The latent size follows the downsampling factor used inside the 10K SNP VAE.
latent_size = int((df_test_noname.shape[1]+1)/(2**12))

vae = VAE(data_shape=df_test_noname.shape[1], latent_size=latent_size, channels=channels, noise_dim = noise_dim, alph=alph)
if (device.type == 'cuda') and (gpu > 1):
    vae = nn.DataParallel(vae, list(range(gpu)))
vae.to(device)


optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

synt_losses = []
test_losses = []

# Synthetic files are named by epoch, e.g. VAE_model_100_output.hapt.
synthetic_pattern = os.path.join(out_dir, f"{model_name}_*_output.hapt")
all_synth_files = glob.glob(synthetic_pattern)

def extract_epoch(filename):
    """Return the epoch encoded in a synthetic output file name."""
    match = re.search(r'_(\d+)_output\.hapt$', filename)
    return int(match.group(1)) if match else -1

# Sort files by extracted epoch
all_synth_files_sorted = sorted(all_synth_files, key=extract_epoch)

for synth_file in all_synth_files_sorted:
    epoch = extract_epoch(synth_file)


    # Each synthetic file is evaluated against the checkpoint saved at the
    # same epoch so losses are aligned with that model state.
    torch.cuda.empty_cache()

    checkpoint_path = f'{out_dir}/{model_name}_{epoch}'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        vae.load_state_dict(checkpoint['VAE'])
        vae.encoder.load_state_dict(checkpoint['Encoder'])
        vae.decoder.load_state_dict(checkpoint['Decoder'])
        print(f"Loaded VAE model from {checkpoint_path}, trained on {epoch} epochs")
        vae.to(device)
    else:
        print ("No checkpoint found, exiting.")
        sys.exit(1)

    vae.eval()

    # Train losses are used as the positive/member distribution in the
    # reconstruction-loss membership attack.
    train_loss = 0.0
    n_train_samples = 0
    train_all_losses = []
    with torch.no_grad():
        for i in range(0, train_x.shape[0], batch_size):
            batch = train_x[i:i+batch_size]
            train_x_hat = vae(batch)
            batch_loss = F.binary_cross_entropy(train_x_hat, batch, reduction="sum") #+ vae.encoder.kl
            train_loss += batch_loss.item()
            n_train_samples += batch.shape[0]

            recon = F.binary_cross_entropy(train_x_hat, batch, reduction="none")  
            # shape: (batch_size, num_features)
            recon_per_sample = recon.sum(dim=(1,2))  
            # shape: (batch_size,)
            # z = vae.encoder(batch)
            train_error_per_sample = recon_per_sample #+ vae.encoder.kl       
            train_all_losses.extend(train_error_per_sample.cpu().tolist())

            
    train_avg_loss = train_loss / n_train_samples
    

    # Test losses are used as the negative/non-member distribution.
    test_loss = 0.0
    n_test_samples = 0
    test_all_losses = []
    with torch.no_grad():
        for i in range(0, test_x.shape[0], batch_size):
            batch = test_x[i:i+batch_size]
            test_x_hat = vae(batch)
            batch_loss = F.binary_cross_entropy(test_x_hat, batch, reduction="sum") #+ vae.encoder.kl
            test_loss += batch_loss.item()
            n_test_samples += batch.shape[0]
            recon = F.binary_cross_entropy(test_x_hat, batch, reduction="none")  
            # shape: (batch_size, num_features)
            recon_per_sample = recon.sum(dim=(1,2))  
            # shape: (batch_size,)
            # z = vae.encoder(batch)
            test_error_per_sample = recon_per_sample #+ vae.encoder.kl           

            test_all_losses.extend(test_error_per_sample.cpu().tolist())


    test_avg_loss = test_loss / n_test_samples

    # Load the synthetic genomes produced by this epoch and compute their
    # average reconstruction loss. This average becomes the attack threshold.
    df_synt = pd.read_hdf(synth_file, key="df1")
    df_synt = df_synt.sample(frac=1).reset_index(drop=True)
    df_synt_noname = df_synt.drop(df_synt.columns[0:2], axis=1)
    df_synt_noname = df_synt_noname.values
    synt_x = torch.tensor(df_synt_noname, device=device).float()
    synt_x = synt_x.reshape(synt_x.shape[0], 1, synt_x.shape[1])

    synt_loss = 0.0
    n_synt_samples = 0
    with torch.no_grad():
        for i in range(0, synt_x.shape[0], batch_size):
            batch = synt_x[i:i+batch_size]
            synt_x_hat = vae(batch)
            batch_loss = F.binary_cross_entropy(synt_x_hat, batch, reduction="sum") #+ vae.encoder.kl
            synt_loss += batch_loss.item()
            n_synt_samples += batch.shape[0]
    synt_avg_loss = synt_loss / n_synt_samples

    # Samples with reconstruction loss below the synthetic average are treated
    # as predicted members.
    train_predicted_member = np.array(train_all_losses) < synt_avg_loss
    test_predicted_member = np.array(test_all_losses) < synt_avg_loss

    Loss_TPR = np.mean(train_predicted_member)
    Loss_FPR = np.mean(test_predicted_member)
    Loss_balanced_accuracy = (Loss_TPR + (1 - Loss_FPR)) / 2
    Loss_privacy_loss_threshold = Loss_TPR - Loss_FPR


    if write_header:
        # Create a new file (overwrite if exists) and write header
        with open(loss_file, 'w') as f:
            f.write('epoch,train_loss,test_loss,synt_loss,train_avg_loss,test_avg_loss,synt_avg_loss,Loss_TPR,Loss_FPR,Loss_balanced_accuracy,Loss_privacy_loss_threshold\n')
        write_header = False

    # Append results to the file
    with open(loss_file, 'a') as f:
        f.write(f'{epoch},{train_loss:.6f},{test_loss:.6f},{synt_loss:.6f},{train_avg_loss:.6f},{test_avg_loss:.6f},{synt_avg_loss:.6f},{Loss_TPR:.6f},{Loss_FPR:.6f},{Loss_balanced_accuracy:.6f},{Loss_privacy_loss_threshold:.6f}\n')

    print(f"Epoch: {epoch}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, Synth Loss: {synt_loss:.6f}, Train Avg Loss: {train_avg_loss:.6f}, Test Avg Loss: {test_avg_loss:.6f}, Synth Avg Loss: {synt_avg_loss:.6f}, Loss TPR: {Loss_TPR:.6f}, Loss FPR: {Loss_FPR:.6f}, Loss Balanced Accuracy: {Loss_balanced_accuracy:.6f}, Loss Privacy Loss Threshold: {Loss_privacy_loss_threshold:.6f}")
