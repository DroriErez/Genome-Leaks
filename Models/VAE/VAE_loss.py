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
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from AA_Simulation.measurements import calc_AA


train_inpt = "Data/1000G_real_genomes/10K_SNP_1000G_real_PADDED_train.hapt" #hdf format input file

test_inpt = "Data/1000G_real_genomes/10K_SNP_1000G_real_PADDED_test.hapt" #hdf format input file

eval_inpt = "Data/1000G_real_genomes/10K_SNP_1000G_real_PADDED_eval.hapt" #hdf format input file

models_dir = "models/VAE/checkpoints"
out_dir = "models/VAE/output"

loss_file = f"{out_dir}/VAE_loss_over_epochs.csv"
model_name = "VAE_model"
epochs = 10001
lr = 0.001
batch_size = 8
channels = 8
gpu = 1 #number of GPUs
alph = 0.001 #alpha value for LeakyReLU
save_that = 100 #epoch interval for saving outputs
AA_SAMPLE_SIZE = 100 #number of random samples to use for AA metrics
noise_dim = 1 #dimension of latent_space

write_header = True

device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu > 0) else "cpu")

df_train = pd.read_csv(train_inpt, sep=' ', header=None)
df_train = df_train.sample(frac=1).reset_index(drop=True)
df_train_noname = df_train.drop(df_train.columns[0:2], axis=1)
df_train_noname = df_train_noname.values
train_x = torch.tensor(df_train_noname, device=device).float()
train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1])

df_test = pd.read_csv(test_inpt, sep = ' ', header=None)
df_test = df_test.sample(frac=1).reset_index(drop=True)
df_test_noname = df_test.drop(df_test.columns[0:2], axis=1)
df_test_noname = df_test_noname.values
test_x = torch.tensor(df_test_noname, device=device).float()
test_x = test_x.reshape(test_x.shape[0], 1, test_x.shape[1])

df_eval = pd.read_csv(eval_inpt, sep=' ', header=None)
df_eval = df_eval.sample(frac=1).reset_index(drop=True)
df_eval_noname = df_eval.drop(df_eval.columns[0:2], axis=1)
df_eval_noname = df_eval_noname.values
eval_x = torch.tensor(df_eval_noname, device=device).float()
eval_x = eval_x.reshape(eval_x.shape[0], 1, eval_x.shape[1])

latent_size = int((df_test_noname.shape[1]+1)/(2**12))

vae = VAE(data_shape=df_test_noname.shape[1], latent_size=latent_size, channels=channels, noise_dim = noise_dim, alph=alph)
if (device.type == 'cuda') and (gpu > 1):
    vae = nn.DataParallel(vae, list(range(gpu)))
vae.to(device)


optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

synt_losses = []
test_losses = []

synthetic_pattern = os.path.join(models_dir, f"{model_name}_*_output.hapt")
all_synth_files = glob.glob(synthetic_pattern)

def extract_epoch(filename):
    match = re.search(r'_(\d+)_output\.hapt$', filename)
    return int(match.group(1)) if match else -1

# Sort files by extracted epoch
all_synth_files_sorted = sorted(all_synth_files, key=extract_epoch)

def call_AA_dist_metrics(training_points, test_points, synth_points):

        AAtr, real2real_dists_tr, real2synth_dists_tr, synth2synth_dists_tr = calc_AA(training_points, synth_points)
        AAte, real2real_dists_te, real2synth_dists_te, synth2synth_dists_te = calc_AA(test_points, synth_points)

        privacy_loss = AAte - AAtr

        return AAtr, AAte, privacy_loss

def random_sample_points(points, sample_size):
    sample_size = min(sample_size, points.shape[0])
    sample_indices = np.random.choice(points.shape[0], size=sample_size, replace=False)
    return points[sample_indices]

def evaluate_losses(model, data_x, batch_size, n_repeats=3):
    total_loss = 0.0
    n_samples = 0
    all_losses = []
    n_repeats = max(1, int(n_repeats))

    with torch.no_grad():
        for i in range(0, data_x.shape[0], batch_size):
            batch = data_x[i:i+batch_size]
            repeated_losses = []

            for _ in range(n_repeats):
                batch_x_hat = model(batch)
                recon = F.binary_cross_entropy(batch_x_hat, batch, reduction="none")
                recon_per_sample = recon.sum(dim=(1, 2))  #+ model.encoder.kl
                repeated_losses.append(recon_per_sample)

            avg_recon_per_sample = torch.stack(repeated_losses).mean(dim=0)
            total_loss += avg_recon_per_sample.sum().item()
            n_samples += batch.shape[0]
            all_losses.extend(avg_recon_per_sample.cpu().tolist())

    avg_loss = total_loss / n_samples
    return total_loss, avg_loss, all_losses

for synth_file in all_synth_files_sorted:
    epoch = extract_epoch(synth_file)


    ## Outputs for assessment at every "save_that" epoch
    torch.cuda.empty_cache()

    checkpoint_path = f'{models_dir}/{model_name}_{epoch}'
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

    train_loss, train_avg_loss, train_all_losses = evaluate_losses(vae, train_x, batch_size)
    test_loss, test_avg_loss, test_all_losses = evaluate_losses(vae, test_x, batch_size)
    eval_loss, eval_avg_loss, eval_all_losses = evaluate_losses(vae, eval_x, batch_size)

    df_synt = pd.read_hdf(synth_file, key="df1")
    df_synt = df_synt.sample(frac=1).reset_index(drop=True)
    df_synt_noname = df_synt.drop(df_synt.columns[0:2], axis=1)
    df_synt_noname = df_synt_noname.values
    synt_x = torch.tensor(df_synt_noname, device=device).float()
    synt_x = synt_x.reshape(synt_x.shape[0], 1, synt_x.shape[1])

    synt_loss, synt_avg_loss, _ = evaluate_losses(vae, synt_x, batch_size)
    AA_train_sample = random_sample_points(df_train_noname, AA_SAMPLE_SIZE)
    AA_test_sample = random_sample_points(df_test_noname, AA_SAMPLE_SIZE)
    AA_synt_sample = random_sample_points(df_synt_noname, AA_SAMPLE_SIZE)
    AA_train, AA_test, AA_privacy_loss = call_AA_dist_metrics(AA_train_sample, AA_test_sample, AA_synt_sample)

    train_predicted_member = np.array(train_all_losses) < synt_avg_loss
    test_predicted_member = np.array(test_all_losses) < synt_avg_loss

    Loss_TPR = np.mean(train_predicted_member)
    Loss_FPR = np.mean(test_predicted_member)
    Loss_balanced_accuracy = (Loss_TPR + (1 - Loss_FPR)) / 2
    Loss_privacy_loss_threshold = Loss_TPR - Loss_FPR

    eval_1pct_threshold = np.percentile(eval_all_losses, 1)
    eval_threshold_train_predicted_member = np.array(train_all_losses) < eval_1pct_threshold
    eval_threshold_test_predicted_member = np.array(test_all_losses) < eval_1pct_threshold
    Eval_1pct_Loss_TPR = np.mean(eval_threshold_train_predicted_member)
    Eval_1pct_Loss_FPR = np.mean(eval_threshold_test_predicted_member)
    Eval_1pct_Loss_balanced_accuracy = (Eval_1pct_Loss_TPR + (1 - Eval_1pct_Loss_FPR)) / 2
    Eval_1pct_Loss_privacy_loss_threshold = Eval_1pct_Loss_TPR - Eval_1pct_Loss_FPR


    if write_header:
        # Create a new file (overwrite if exists) and write header
        with open(loss_file, 'w') as f:
            f.write('epoch,train_loss,test_loss,eval_loss,synt_loss,train_avg_loss,test_avg_loss,eval_avg_loss,synt_avg_loss,Loss_TPR,Loss_FPR,Loss_balanced_accuracy,Loss_privacy_loss_threshold,eval_1pct_threshold,Eval_1pct_Loss_TPR,Eval_1pct_Loss_FPR,Eval_1pct_Loss_balanced_accuracy,Eval_1pct_Loss_privacy_loss_threshold,AA_train,AA_test,AA_privacy_loss\n')
        write_header = False

    # Append results to the file
    with open(loss_file, 'a') as f:
        f.write(f'{epoch},{train_loss:.6f},{test_loss:.6f},{eval_loss:.6f},{synt_loss:.6f},{train_avg_loss:.6f},{test_avg_loss:.6f},{eval_avg_loss:.6f},{synt_avg_loss:.6f},{Loss_TPR:.6f},{Loss_FPR:.6f},{Loss_balanced_accuracy:.6f},{Loss_privacy_loss_threshold:.6f},{eval_1pct_threshold:.6f},{Eval_1pct_Loss_TPR:.6f},{Eval_1pct_Loss_FPR:.6f},{Eval_1pct_Loss_balanced_accuracy:.6f},{Eval_1pct_Loss_privacy_loss_threshold:.6f},{AA_train:.6f},{AA_test:.6f},{AA_privacy_loss:.6f}\n')

    print(f"Epoch: {epoch}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, Eval Loss: {eval_loss:.6f}, Synth Loss: {synt_loss:.6f}, Train Avg Loss: {train_avg_loss:.6f}, Test Avg Loss: {test_avg_loss:.6f}, Eval Avg Loss: {eval_avg_loss:.6f}, Synth Avg Loss: {synt_avg_loss:.6f}, Loss TPR: {Loss_TPR:.6f}, Loss FPR: {Loss_FPR:.6f}, Loss Balanced Accuracy: {Loss_balanced_accuracy:.6f}, Loss Privacy Loss Threshold: {Loss_privacy_loss_threshold:.6f}, Eval 1pct Threshold: {eval_1pct_threshold:.6f}, Eval 1pct Loss TPR: {Eval_1pct_Loss_TPR:.6f}, Eval 1pct Loss FPR: {Eval_1pct_Loss_FPR:.6f}, Eval 1pct Loss Balanced Accuracy: {Eval_1pct_Loss_balanced_accuracy:.6f}, Eval 1pct Loss Privacy Loss Threshold: {Eval_1pct_Loss_privacy_loss_threshold:.6f}, AA Train: {AA_train:.6f}, AA Test: {AA_test:.6f}, AA Privacy Loss: {AA_privacy_loss:.6f}")
