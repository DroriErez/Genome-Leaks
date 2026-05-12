import sys
import os
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from pca_plot_genomes import pca_plot
from numpy.random import randint
import torch.backends.cudnn as cudnn

# Code for WGAN training on 65k data
#from models_65K import * 
# inpt = "1000G_real_genomes/65K_SNP_1000G_real_train_small.hapt" #hapt format input file
# eval_inpt = "1000G_real_genomes/65K_SNP_1000G_real_eval_small.hapt" #hdf format input file
# latent_depth_factor = 14 #14 for 65535 SNP data and 12 for 16383 zero padded SNP data

# Code for WGAN training on 10k padded data
from models_10K import * #for 16383 zero padded SNP data
inpt = "Data/1000G_real_genomes/10K_SNP_1000G_real_PADDED_train.hapt" #hdf format input file
eval_inpt = "Data/1000G_real_genomes/10K_SNP_1000G_real_PADDED_eval.hapt" #hdf format input file
latent_depth_factor = 12 #14 for 65535 SNP data and 12 for 16383 zero padded SNP data

## Set seed for reproducibility
manualSeed = 9
random.seed(manualSeed)
torch.manual_seed(manualSeed)
print("Random Seed: ", manualSeed)


out_dir = "./models/WGAN/checkpoints"
model_name = "WGAN_model"

alph = 0.01 #alpha value for LeakyReLU
g_learn = 0.0005 #generator learning rate
d_learn = 0.0005 #discriminator learning rate
epochs = 10001
batch_size = 8 #96
channels = 10 #channel multiplier which dictates the number of channels for all layers
ag_size = 500 #number of artificial genomes (haplotypes) to be created
gpu = 1 #number of GPUs
save_that = 100 #epoch interval for saving outputs
pack_m = 3 #packing amount for the critic
critic_iter = 10 #number of times critic is trained for every generator training
label_noise = 1 #noise for real labels (1: noise, 0: no noise)
noise_dim = 2 #dimension of noise for each noise vector

device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu > 0) else "cpu") #set device to cpu or gpu

print (f"Using device: {device}, with {gpu} GPU(s)")

def read_hapt_matrix(path, pca_rows=0, seed=None):
    with open(path, "r") as handle:
        first_line = handle.readline()
        if not first_line:
            raise ValueError(f"{path} is empty")
        first_values = first_line.split(maxsplit=2)
        if len(first_values) < 3:
            raise ValueError(f"{path} must contain two metadata columns and genotype columns")
        n_cols = np.fromstring(first_values[2], dtype=np.uint8, sep=" ").shape[0]
        n_rows = 1 + sum(1 for line in handle if line.strip())

    genomes = np.empty((n_rows, n_cols), dtype=np.uint8)
    metadata = []
    with open(path, "r") as handle:
        row_idx = 0
        for line in handle:
            if not line.strip():
                continue
            values = line.split(maxsplit=2)
            if len(values) < 3:
                raise ValueError(f"Missing genotype columns in {path} at data row {row_idx}")
            genotype = np.fromstring(values[2], dtype=np.uint8, sep=" ")
            if genotype.shape[0] != n_cols:
                raise ValueError(
                    f"Unexpected column count in {path} at data row {row_idx}: "
                    f"expected {n_cols} genotypes, got {genotype.shape[0]}"
                )
            if pca_rows:
                metadata.append(values[:2])
            genomes[row_idx] = genotype
            row_idx += 1

    rng = np.random.default_rng(seed)
    pca_idx = rng.permutation(n_rows)[:min(pca_rows, n_rows)] if pca_rows else []
    if pca_rows:
        pca_metadata = np.asarray(metadata, dtype=object)[pca_idx]
        pca_df = pd.DataFrame(genomes[pca_idx].astype(np.uint8, copy=False))
        pca_df.insert(loc=0, column="Type", value=pca_metadata[:, 0])
        pca_df.insert(loc=1, column="ID", value=pca_metadata[:, 1])
    else:
        pca_df = None

    return genomes, pca_df

## Prepare the training data
df_noname, df = read_hapt_matrix(inpt, pca_rows=ag_size, seed=manualSeed)
#df_noname = df_noname - np.random.uniform(0,0.1, size=(df_noname.shape[0], df_noname.shape[1]))
dataloader = torch.utils.data.DataLoader(
    torch.from_numpy(df_noname),
    batch_size=batch_size,
    shuffle=True,
    pin_memory=(device.type == "cuda"),
    drop_last=True
)

df_eval_noname, _ = read_hapt_matrix(eval_inpt, seed=manualSeed)
eval_x = torch.from_numpy(df_eval_noname)


latent_size = int((df_noname.shape[1]+1)/(2**latent_depth_factor)) #set the latent_size

## Create the generator
netG = ConvGenerator(latent_size=latent_size, data_shape=df_noname.shape[1], gpu=gpu, device=device, channels=channels, noise_dim=noise_dim, alph=alph)
netG = netG.float()
if (device.type == 'cuda') and (gpu > 1):
    netG = nn.DataParallel(netG, list(range(gpu)))
netG.to(device)

## Create the critic
netC = ConvDiscriminator(data_shape=df_noname.shape[1], latent_size=latent_size, gpu=gpu, pack_m = pack_m, device=device, channels=channels, alph=alph).to(device)
netC = netC.float()
if (device.type == 'cuda') and (gpu > 1):
    netC = nn.DataParallel(netC, list(range(gpu)))
netC.to(device)

## Optimizers for generator and critic
c_optimizer = torch.optim.Adam(netC.parameters(), lr=d_learn, betas=(0.5, 0.9))
g_optimizer = torch.optim.Adam(netG.parameters(), lr=g_learn, betas=(0.5, 0.9))

label_fake = torch.tensor(1, dtype=torch.float).to(device)
label_real = label_fake * -1
losses = []
use_label_noise = label_noise != 0

def clean_cuda_memory():
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

## Noise generator function to be used to provide input noise vectors
def noise_generator(size, noise_count, noise_dim, device):
    noise_list = []
    for i in range(2,noise_count*2+1,2):
        noise = torch.normal(mean=0, std=1, size=(size, noise_dim, latent_size*(2**i)-1), device = device)
        noise_list.append(noise)
    return noise_list

checkpoint_path = f'{out_dir}/{model_name}_last_model'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    netG.load_state_dict(checkpoint['Generator'])
    netC.load_state_dict(checkpoint['Critic'])
    if 'G_optimizer' in checkpoint:
        g_optimizer.load_state_dict(checkpoint['G_optimizer'])
    if 'C_optimizer' in checkpoint:
        c_optimizer.load_state_dict(checkpoint['C_optimizer'])
    start_epoch = checkpoint.get('epoch', 0)+1
    print(f"Loaded WGAN model from {checkpoint_path}, starting from epoch {start_epoch}")
else:
    start_epoch = 0


## Training Loop
print("Starting Training Loop...")
print(
    "Training parameters: "
    f"input_snps={df_noname.shape[1]}, "
    f"train_samples={df_noname.shape[0]}, "
    f"eval_samples={df_eval_noname.shape[0]}, "
    f"batch_size={batch_size}, "
    f"epochs={epochs}, "
    f"start_epoch={start_epoch}, "
    f"latent_size={latent_size}, "
    f"latent_depth_factor={latent_depth_factor}, "
    f"channels={channels}, "
    f"noise_dim={noise_dim}, "
    f"pack_m={pack_m}, "
    f"critic_iter={critic_iter}, "
    f"g_learn={g_learn}, "
    f"d_learn={d_learn}, "
    f"label_noise={label_noise}, "
    f"save_that={save_that}, "
    f"ag_size={ag_size}, "
    f"device={device}"
)
start_time = time.time()
for epoch in range(start_epoch, epochs):
    #c_loss_real = 0
    #c_loss_fake = 0
    train_critic_difference = 0.0
    n_critic_samples = 0
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - epoch number {epoch}")

    b = 0
    data_iter = [iter(dataloader)]

    def next_real_batch():
        try:
            batch = next(data_iter[0])
        except StopIteration:
            data_iter[0] = iter(dataloader)
            batch = next(data_iter[0])
        return batch.to(device, non_blocking=True).float()

    while b < len(dataloader):
        for param in netC.parameters():
            param.requires_grad = True

        #Update Critic
        for n_critic in range(critic_iter):

            netC.zero_grad(set_to_none=True)

            X_batch_real = next_real_batch()
            X_batch_real = torch.reshape(X_batch_real, (X_batch_real.shape[0], 1, X_batch_real.shape[1]))

            if pack_m > 1:
                for i in range(pack_m-1):
                    temp_batch = next_real_batch()
                    temp_batch = torch.reshape(temp_batch, (temp_batch.shape[0], 1, temp_batch.shape[1]))
                    X_batch_real = torch.cat((X_batch_real, temp_batch), 1)
                    del temp_batch
            b += 1

            #Train Critic with real samples
            c_optimizer.zero_grad(set_to_none=True)
            c_loss_real = netC(X_batch_real)
            c_loss_real = c_loss_real.mean()

            if use_label_noise:
                label_noise_value = torch.tensor(random.uniform(0, 0.1), dtype=torch.float, device=device)
                real_loss_weight = label_real + label_noise_value
            else:
                real_loss_weight = label_real

            with torch.no_grad():
                for i in range(pack_m):
                    latent_samples = torch.normal(mean=0, std=1, size=(batch_size, noise_dim, latent_size), device=device) #create the initial noise to be fed to generator
                    noise_list = noise_generator(batch_size, 6, noise_dim, device)
                    temp = netG(latent_samples, noise_list)
                    if i == 0:
                        X_batch_fake = temp
                    else:
                        X_batch_fake = torch.cat((X_batch_fake, temp), 1)

            c_loss_fake = netC(X_batch_fake.detach())
            c_loss_fake = c_loss_fake.mean()

            batch_critic_difference = c_loss_real.item() - c_loss_fake.item()
            train_critic_difference += batch_critic_difference * X_batch_real.shape[0]
            n_critic_samples += X_batch_real.shape[0]

            #Train with gradient penalty
            gp = gradient_penalty(netC, X_batch_real, X_batch_fake.detach(), device)
            c_loss = c_loss_fake * label_fake + c_loss_real * real_loss_weight + gp
            c_loss.backward()
            c_loss_value = c_loss.item()
            c_optimizer.step()
            c_optimizer.zero_grad(set_to_none=True)
            del X_batch_real, X_batch_fake, latent_samples, noise_list, temp
            del c_loss_real, c_loss_fake, gp, c_loss, batch_critic_difference, real_loss_weight
            if use_label_noise:
                del label_noise_value
            clean_cuda_memory()


        for param in netC.parameters():
            param.requires_grad = False

        #Update G network
        g_optimizer.zero_grad(set_to_none=True)
        netG.zero_grad(set_to_none=True)
        for i in range(pack_m):
            latent_samples = torch.normal(mean=0, std=1, size=(batch_size, noise_dim, latent_size), device=device) #create the initial noise to be fed to generator
            noise_list = noise_generator(batch_size, 6, noise_dim, device)
            temp = netG(latent_samples, noise_list)
            if i == 0:
                X_batch_fake = temp
            else:
                X_batch_fake = torch.cat((X_batch_fake, temp), 1)

        g_loss = netC(X_batch_fake)
        g_loss = g_loss.mean()
        g_loss.backward(label_real)
        g_loss_value = g_loss.item()
        g_optimizer.step()
        g_optimizer.zero_grad(set_to_none=True)

        # Save Losses for plotting later
        #losses.append((c_loss.item(), g_loss.item()))
        losses.append((round(c_loss_value, 3), (round(g_loss_value, 3))))
        del X_batch_fake, latent_samples, noise_list, temp, g_loss
        clean_cuda_memory()

    ## Outputs for assessment at every "save_that" epoch
    if epoch%save_that == 0 or epoch == epochs:
        clean_cuda_memory()
        torch.save({
        'Generator': netG.state_dict(),
        'Critic': netC.state_dict(),
        'G_optimizer': g_optimizer.state_dict(),
        'C_optimizer': c_optimizer.state_dict()},
        f'{out_dir}/{model_name}_{epoch}')

        netG.eval()
        latent_samples = torch.normal(mean=0, std=1, size=(ag_size, noise_dim, latent_size), device=device) #create the initial noise to be fed to generator
        noise_list = noise_generator(ag_size, 6, noise_dim, device)
        with torch.no_grad():
            generated_genomes = netG(latent_samples, noise_list)
        generated_genomes = generated_genomes.cpu().detach().numpy()
        del latent_samples, noise_list
        clean_cuda_memory()
        generated_genomes[generated_genomes < 0] = 0
        generated_genomes = np.rint(generated_genomes)
        generated_genomes_df = pd.DataFrame(np.reshape(generated_genomes, (ag_size, generated_genomes.shape[2])))
        generated_genomes_df = generated_genomes_df.astype(int)
        gen_names = list()

        for i in range(0,len(generated_genomes_df)):
            gen_names.append('AG'+str(i))
        generated_genomes_df.insert(loc=0, column='Type', value="AG")
        generated_genomes_df.insert(loc=1, column='ID', value=gen_names)
        generated_genomes_df.columns = list(range(generated_genomes_df.shape[1]))
        df.columns = list(range(df.shape[1]))

        #Output AGs in hapt or hdf formats
        #generated_genomes_df.to_csv(f'{out_dir}/{epoch}_output.hapt', sep=" ", header=False, index=False)
        generated_genomes_df.to_hdf(f'{out_dir}/{epoch}_output.hapt', key="df1", mode="w")

        #Output lossess
        pd.DataFrame(losses).to_csv(f'{out_dir}/{epoch}_losses.txt', sep=" ", header=False, index=False)
        fig, ax = plt.subplots()
        plt.plot(np.array([losses]).T[0], label='Critic')
        plt.plot(np.array([losses]).T[1], label='Generator')
        plt.title("Training Losses")
        plt.legend()
        fig.savefig(f'{out_dir}/{epoch}_loss.pdf', format='pdf')
        plt.close(fig)

        #Plot PCA
        pca_plot(df, generated_genomes_df, epoch, dir=out_dir)

        # Evaluate critic difference in batches to minimize memory usage
        eval_critic_difference = 0.0
        n_eval_samples = 0
        if eval_x.shape[1] == df_noname.shape[1]:
            netC.eval()
            batch_eval_size = 128  # Adjust as needed for your GPU/CPU
            with torch.no_grad():
                for i in range(0, len(eval_x), batch_eval_size):
                    X_batch_real = eval_x[i:i+batch_eval_size].to(device, non_blocking=True).float()
                    X_batch_real = X_batch_real.reshape(X_batch_real.shape[0], 1, X_batch_real.shape[1])
                    current_batch_size = X_batch_real.shape[0]

                    if current_batch_size == 0:
                        continue

                    if pack_m > 1:
                        for j in range(pack_m-1):
                            indices = torch.randint(0, len(eval_x), (current_batch_size,))
                            temp_batch = eval_x[indices].to(device, non_blocking=True).float()
                            temp_batch = temp_batch.reshape(temp_batch.shape[0], 1, temp_batch.shape[1])
                            X_batch_real = torch.cat((X_batch_real, temp_batch), 1)
                            del temp_batch

                    for j in range(pack_m):
                        latent_samples = torch.normal(mean=0, std=1, size=(current_batch_size, noise_dim, latent_size), device=device)
                        noise_list = noise_generator(current_batch_size, 6, noise_dim, device)
                        temp = netG(latent_samples, noise_list)
                        if j == 0:
                            X_batch_fake = temp
                        else:
                            X_batch_fake = torch.cat((X_batch_fake, temp), 1)

                    eval_loss_real = netC(X_batch_real).mean()
                    eval_loss_fake = netC(X_batch_fake).mean()
                    batch_critic_difference = eval_loss_real.item() - eval_loss_fake.item()
                    eval_critic_difference += batch_critic_difference * current_batch_size
                    n_eval_samples += current_batch_size
                    del X_batch_real, X_batch_fake, latent_samples, noise_list, temp
                    del eval_loss_real, eval_loss_fake, batch_critic_difference
                    clean_cuda_memory()
        else:
            print(f"Skipping eval critic difference: eval SNP count {eval_x.shape[1]} does not match train SNP count {df_noname.shape[1]}")

        train_avg_critic_difference = train_critic_difference / n_critic_samples
        eval_avg_critic_difference = eval_critic_difference / n_eval_samples if n_eval_samples > 0 else np.nan

        critic_difference_file = f'{out_dir}/{model_name}_epoch_critic_difference.csv'
        write_header = not os.path.exists(critic_difference_file)
        with open(critic_difference_file, 'a') as f:
            if write_header:
                f.write('epoch,train_critic_difference,eval_critic_difference\n')
            f.write(f'{epoch},{train_avg_critic_difference:.6f},{eval_avg_critic_difference:.6f}\n')

        print(f"Epoch: {epoch}, Train Critic Difference: {train_avg_critic_difference:.6f}, Eval Critic Difference: {eval_avg_critic_difference:.6f}")

        torch.save({
        'epoch': epoch,
        'Generator': netG.state_dict(),
        'Critic': netC.state_dict(),
        'G_optimizer': g_optimizer.state_dict(),
        'C_optimizer': c_optimizer.state_dict()},
        f'{out_dir}/{model_name}_last_model')

        netG.train()
        netC.train()
        del generated_genomes, generated_genomes_df
        clean_cuda_memory()
print("--- %s seconds ---" % (time.time() - start_time))
