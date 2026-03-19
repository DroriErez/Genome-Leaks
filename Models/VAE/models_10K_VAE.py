import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Block(nn.Module):
    def __init__(self, channels, mult, block_type, sampling, noise_dim=None, alph=0.01):
        super().__init__()

        if block_type=="g" and sampling == 1:
            self.block = nn.Sequential(
                nn.Conv1d(channels*mult + 1, channels*(mult-2), 3, stride = 1, padding=1, bias=False),
                nn.BatchNorm1d(channels*(mult-2)),
                nn.LeakyReLU(alph),

                nn.ConvTranspose1d(channels*(mult-2), channels*(mult-4), 3, stride = 2, padding=0, bias=False),
                nn.BatchNorm1d(channels*(mult-4)),
                nn.LeakyReLU(alph),

                nn.Conv1d(channels*(mult-4), channels*(mult-6), 3, stride = 1, padding=1, bias=False),
                nn.BatchNorm1d(channels*(mult-6)),
                nn.LeakyReLU(alph),

                nn.ConvTranspose1d(channels*(mult-6), channels*(mult-8), 3, stride = 2, padding=0, bias=False),
                nn.BatchNorm1d(channels*(mult-8)),
                nn.LeakyReLU(alph))

        elif block_type=="d" and sampling == -1:
            self.block = nn.Sequential(
                nn.Conv1d(channels*mult + 1, channels*(mult+2), 3, stride = 1, padding=1, bias=False),
                nn.BatchNorm1d(channels*(mult+2)),
                nn.LeakyReLU(alph),

                nn.Conv1d(channels*(mult+2), channels*(mult+4), 3, stride = 2, padding=0, bias=False),
                nn.BatchNorm1d(channels*(mult+4)),
                nn.LeakyReLU(alph),

                nn.Conv1d(channels*(mult+4), channels*(mult+6), 3, stride = 1, padding=1, bias=False),
                nn.BatchNorm1d(channels*(mult+6)),
                nn.LeakyReLU(alph),

                nn.Conv1d(channels*(mult+6), channels*(mult+8), 3, stride = 2, padding=0, bias=False),
                nn.BatchNorm1d(channels*(mult+8)),
                nn.LeakyReLU(alph))

        elif block_type=="d" and sampling == 0:
            self.block = nn.Sequential(
                nn.Conv1d(channels*mult, channels*mult, 3, stride = 1, padding=1, bias=False),
                nn.BatchNorm1d(channels*mult),
                nn.LeakyReLU(alph),

                nn.Conv1d(channels*mult, channels*mult, 3, stride = 1, padding=1, bias=False),
                nn.BatchNorm1d(channels*mult),
                nn.LeakyReLU(alph))

        elif block_type=="g" and sampling == 0:
            self.block = nn.Sequential(
                nn.Conv1d(channels*mult, channels*mult, 3, stride = 1, padding=1, bias=False),
                nn.BatchNorm1d(channels*mult),
                nn.LeakyReLU(alph),

                nn.Conv1d(channels*mult, channels*mult, 3, stride = 1, padding=1, bias=False),
                nn.BatchNorm1d(channels*mult),
                nn.LeakyReLU(alph))

    def forward(self, x):
        return self.block(x)

class Decoder(nn.Module):
    def __init__(self, latent_size, data_shape, channels, noise_dim, alph):
        super(Decoder, self).__init__()

        #parameters initialization
        self.channels = channels
        self.latent_size = latent_size
        self.alph = alph
        self.data_shape = data_shape
        self.noise_dim = noise_dim


        self.ms_vars = nn.ParameterList()
        self.ms_vars.append(nn.Parameter(torch.normal(mean=0, std=1, size=(1, latent_size))))
        for i in range(2,14,2):
            self.ms_vars.append(nn.Parameter(torch.normal(mean=0, std=1, size=(1, latent_size*(2**i)-1))))

        self.block1 = nn.Sequential(
                    nn.Conv1d(noise_dim + 1, self.channels*44, 3, stride = 1, padding=1, bias=False),
                    nn.BatchNorm1d(self.channels*44),
                    nn.LeakyReLU(alph),

                    nn.ConvTranspose1d(self.channels*44, self.channels*42, 3, stride = 2, padding=1, bias=False),
                    nn.BatchNorm1d(self.channels*42),
                    nn.LeakyReLU(alph),

                    nn.Conv1d(self.channels*42, self.channels*40, 3, stride = 1, padding=1, bias=False),
                    nn.BatchNorm1d(self.channels*40),
                    nn.LeakyReLU(alph),

                    nn.ConvTranspose1d(self.channels*40, self.channels*38, 3, stride = 2, padding=0, bias=False),
                    nn.BatchNorm1d(self.channels*38),
                    nn.LeakyReLU(alph),
        )
        self.block2 = Block(channels = self.channels, mult = 38, block_type = "g", sampling = 1, noise_dim = self.noise_dim)
        self.block3 = Block(channels = self.channels, mult = 30, block_type = "g", sampling = 0, noise_dim = self.noise_dim)
        self.block4 = Block(channels = self.channels, mult = 30, block_type = "g", sampling = 1, noise_dim = self.noise_dim)
        self.block5 = Block(channels = self.channels, mult = 22, block_type = "g", sampling = 1, noise_dim = self.noise_dim)
        self.block6 = Block(channels = self.channels, mult = 14, block_type = "g", sampling = 0, noise_dim = self.noise_dim)
        self.block7 = Block(channels = self.channels, mult = 14, block_type = "g", sampling = 1, noise_dim = self.noise_dim)
        self.block8 = nn.Sequential(
                    nn.Conv1d(self.channels * 6 + 1, self.channels * 4, 3, stride=1, padding=1, bias=False),
                    nn.BatchNorm1d(self.channels*4),
                    nn.LeakyReLU(alph),

                    nn.ConvTranspose1d(self.channels * 4, self.channels * 2, 3, stride=2, padding=0, bias=False),
                    nn.BatchNorm1d(self.channels*2),
                    nn.LeakyReLU(alph),

                    nn.Conv1d(self.channels * 2, self.channels * 1, 3, stride=1, padding=1, bias=False),
                    nn.BatchNorm1d(self.channels*1),
                    nn.LeakyReLU(alph),

                    nn.ConvTranspose1d(self.channels * 1, (self.channels * 1)//2, 3, stride=2, padding=0, bias=False),
                    nn.BatchNorm1d((self.channels * 1)//2),
                    nn.LeakyReLU(alph),
        )
        self.block9 = nn.Sequential(
                    nn.Conv1d((self.channels * 1)//2 + 1, 1, 3, stride=1, padding=1),
                    nn.Sigmoid()
        )


    def forward(self, x):

        batch_size = x.shape[0]
        x = torch.cat((self.ms_vars[0].repeat(batch_size,1)[:,np.newaxis,:], x), 1)
        x = self.block1(x)

        x = torch.cat((self.ms_vars[1].repeat(batch_size,1)[:,np.newaxis,:], x), 1)
        x = self.block2(x)

        res = x
        x = self.block3(x)
        x += res

        x = torch.cat((self.ms_vars[2].repeat(batch_size,1)[:,np.newaxis,:], x), 1)
        x = self.block4(x)

        x = torch.cat((self.ms_vars[3].repeat(batch_size,1)[:,np.newaxis,:], x), 1)
        x = self.block5(x)

        res = x
        x = self.block6(x)
        x += res

        x = torch.cat((self.ms_vars[4].repeat(batch_size,1)[:,np.newaxis,:], x), 1)
        x = self.block7(x)

        x = torch.cat((self.ms_vars[5].repeat(batch_size,1)[:,np.newaxis,:], x), 1)
        x = self.block8(x)

        x = torch.cat((self.ms_vars[6].repeat(batch_size,1)[:,np.newaxis,:], x), 1)
        x = self.block9(x)

        return x



class Encoder(nn.Module):
    def __init__(self, data_shape, latent_size, channels, alph):
        super(Encoder, self).__init__()

        self.alph = alph
        self.data_shape = data_shape
        self.channels = channels
        self.latent_size = latent_size

        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.Norm= torch.distributions.Normal(0, 1)
        self.Norm.loc = self.Norm.loc.to(device)
        self.Norm.scale = self.Norm.scale.to(device)
        self.kl = 0

        self.ms_vars = nn.ParameterList()
        for i in range(12,1,-2):
            self.ms_vars.append(nn.Parameter(torch.normal(mean=0, std=1, size=(1, latent_size*(2**i)-1))))

        self.block1 = nn.Sequential(
                    nn.Conv1d(1 + 1, self.channels * 1, 3, stride=1, padding=1, bias=False),
                    nn.BatchNorm1d(self.channels*1),
                    nn.LeakyReLU(alph),

                    nn.Conv1d(self.channels * 1, self.channels * 2, 3, stride=2, padding=0, bias=False),
                    nn.BatchNorm1d(self.channels*2),
                    nn.LeakyReLU(alph),

                    nn.Conv1d(self.channels * 2, self.channels * 4, 3, stride=1, padding=1, bias=False),
                    nn.BatchNorm1d(self.channels*4),
                    nn.LeakyReLU(alph),

                    nn.Conv1d(self.channels * 4, self.channels * 6, 3, stride=2, padding=0, bias=False),
                    nn.BatchNorm1d(self.channels*6),
                    nn.LeakyReLU(alph),
        )
        self.block2 = Block(channels = self.channels, mult = 6, block_type = "d", sampling = 0)
        self.block3 = Block(channels = self.channels, mult = 6, block_type = "d", sampling = -1)
        self.block4 = Block(channels = self.channels, mult = 14, block_type = "d", sampling = -1)
        self.block5 = Block(channels = self.channels, mult = 22, block_type = "d", sampling = 0)
        self.block6 = Block(channels = self.channels, mult = 22, block_type = "d", sampling = -1)
        self.block7 = Block(channels = self.channels, mult = 30, block_type = "d", sampling = -1)
        self.block8 = nn.Sequential(
                    nn.Conv1d(self.channels * 38 + 1, self.channels * 40, 3, stride=1, padding=1, bias=False),
                    nn.BatchNorm1d(self.channels * 40),
                    nn.LeakyReLU(alph),

                    nn.Conv1d(self.channels * 40, self.channels * 42, 3, stride=2, padding=0, bias=False),
                    nn.BatchNorm1d(self.channels*42),
                    nn.LeakyReLU(alph),

                    nn.Conv1d(self.channels * 42, self.channels * 44, 3, stride=1, padding=1, bias=False),
                    nn.BatchNorm1d(self.channels * 44),
                    nn.LeakyReLU(alph),
        )

        self.block_last_mu = nn.Sequential(
                    #nn.Linear(latent_size, latent_size)
                    nn.Conv1d(self.channels * 44, 1, 3, stride=2, padding=1)


        )

        self.block_last_sigma = nn.Sequential(
                    #nn.Linear(latent_size, latent_size)
                    nn.Conv1d(self.channels * 44, 1, 3, stride=2, padding=1)

        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = torch.cat((self.ms_vars[0].repeat(batch_size,1)[:,np.newaxis,:], x), 1)
        x = self.block1(x)

        res = x
        x = self.block2(x)
        x += res

        x = torch.cat((self.ms_vars[1].repeat(batch_size,1)[:,np.newaxis,:], x), 1)
        x = self.block3(x)

        x = torch.cat((self.ms_vars[2].repeat(batch_size,1)[:,np.newaxis,:], x), 1)
        x = self.block4(x)

        res = x
        x = self.block5(x)
        x += res

        x = torch.cat((self.ms_vars[3].repeat(batch_size,1)[:,np.newaxis,:], x), 1)
        x = self.block6(x)

        x = torch.cat((self.ms_vars[4].repeat(batch_size,1)[:,np.newaxis,:], x), 1)
        x = self.block7(x)

        x = torch.cat((self.ms_vars[5].repeat(batch_size,1)[:,np.newaxis,:], x), 1)
        x = self.block8(x)

        mu =  self.block_last_mu(x)
        sigma = torch.exp(self.block_last_sigma(x))
        z = mu + sigma*self.Norm.sample(mu.shape)
        self.kl = -0.5 * torch.sum(1 + torch.log(sigma**2) - mu**2  - sigma**2)

        return z


class VAE(nn.Module):
    def __init__(self, data_shape, latent_size, channels, noise_dim, alph):
        super(VAE, self).__init__()
        self.encoder = Encoder(data_shape=data_shape, latent_size=latent_size, channels=channels, alph=alph)
        self.decoder = Decoder(latent_size=latent_size, data_shape=data_shape, channels=channels, noise_dim= noise_dim, alph=alph)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
