""" Copying Phil's notebook from  
https://github.com/RomeroLab/promero/blob/main/VAEs/train_VAE.ipynb
and adding weights to the training process """


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import pytorch_lightning as pl

from collections import OrderedDict
from torchtext import vocab
from pytorch_lightning.loggers import CSVLogger
from random import choice
import seaborn as sns


# setup torchtext vocab to map AAs to indices, usage is aa2ind(list(AAsequence))
AAs = 'ACDEFGHIKLMNPQRSTVWY-'
aa2ind = vocab.vocab(OrderedDict([(a, 1) for a in AAs]))
aa2ind.set_default_index(20) # set unknown charcterers to gap

def get_aaindex_embedding(filename="../../protein_utils/data/AA/pca-19.csv",
        ncomp = 8):
    # get aa index embedding
    aaindex = np.array([[float(f) for f in l.split(',')[1:]] for l in 
            open(filename).read().strip().split('\n')[1:]])
    aaindex = (aaindex - aaindex.mean(0))/aaindex.std(0) # standardize
    aaindex = np.vstack([aaindex,np.zeros((1,19))]) # add final row to include gap -
    aaindex = torch.from_numpy(aaindex).float() 
    aaindex = aaindex[:,:ncomp]
    return aaindex

class ProtMSA(torch.utils.data.Dataset):
   
    def __init__(self, MSA, weights=None):
        self.MSA = MSA
        self.weights = weights
        if self.weights is None:
            self.weights = np.ones(len(MSA), dtype=float)
        assert(len(MSA) == len(self.weights))

    def __getitem__(self, idx):
        # index the MSA 
        sequence = torch.tensor(aa2ind(list(self.MSA[idx])))
        return sequence, self.weights[idx]

    def __len__(self):
        return len(self.MSA)

class ProtDataModule(pl.LightningDataModule):
    """A PyTorch Lightning Data Module to handle data splitting"""

    def __init__(self, MSA, batch_size):
        super().__init__()
        self.MSA = MSA
        self.batch_size = batch_size
        train_val_test_split = [0.8, 0.1, 0.1]
        n_train_val_test = np.round(np.array(train_val_test_split)*len(MSA)).astype(int)
        if sum(n_train_val_test)<len(MSA): n_train_val_test[0] += 1 # necesary when round is off by 1
        if sum(n_train_val_test)>len(MSA): n_train_val_test[0] -= 1 
        self.train_idx, self.val_idx, self.test_idx = data_utils.random_split(range(len(MSA)),n_train_val_test)

    def prepare_data(self):
        # prepare_data is called from a single GPU. Do not use it to assign state (self.x = y)
        # use this method to do things that might write to disk or that need to be done only from a single process
        # in distributed settings.
        pass
        
    def setup(self, stage=None):
              
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            train_MSA = [self.MSA[i] for i in self.train_idx]
            self.train_MSA = ProtMSA(train_MSA)
            
            val_MSA = [self.MSA[i] for i in self.val_idx]
            self.val_MSA = ProtMSA(val_MSA)
            
        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            test_MSA = [self.MSA[i] for i in self.test_idx]
            self.test_MSA = ProtMSA(test_MSA)

    def train_dataloader(self):
        return data_utils.DataLoader(self.train_MSA, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return data_utils.DataLoader(self.val_MSA, batch_size=self.batch_size)

    def test_dataloader(self):
        return data_utils.DataLoader(self.test_MSA, batch_size=self.batch_size)


class ConvVAE(pl.LightningModule):
        
    def __init__(self, slen, ks, nlatent, learning_rate=0.0001):
        super().__init__()
      
        ## GENERAL INPUT PARAMETERS
        self.slen = slen # sequence length
        self.ks = ks # kernel size
        self.nlatent = nlatent # num latent vars
        self.learning_rate = learning_rate
    
        ## ENCODER
        aaindex = get_aaindex_embedding(ncomp=8)
        self.embed = nn.Embedding.from_pretrained(aaindex, freeze=False)
        edim = self.embed.embedding_dim # dimensions of AA embedding
        self.enc_conv_1 = torch.nn.Conv1d(in_channels=  edim, out_channels=2*edim, kernel_size=ks)
        self.enc_conv_2 = torch.nn.Conv1d(in_channels=2*edim, out_channels=4*edim, kernel_size=ks) 
        nparam = (slen-2*(ks-1))*(4*edim) # each convolution reduces slen by ks-1, multiply by the # output channels 
        self.z_mean = torch.nn.Linear(nparam,nlatent)
        self.z_log_var = torch.nn.Linear(nparam,nlatent)

        ## DECODER
        self.dec_linear_1 = torch.nn.Linear(nlatent,nparam)
        self.dec_deconv_1 = torch.nn.ConvTranspose1d(in_channels=4*edim, out_channels=2*edim, kernel_size=ks)
        self.dec_deconv_2 = torch.nn.ConvTranspose1d(in_channels=2*edim, out_channels=  edim, kernel_size=ks)
        nembed = self.embed.num_embeddings
        self.rev_embed = torch.nn.Linear(edim,nembed)
        
        # record additional dimensions for reshaping
        self.edim = edim
        self.nparam = nparam
        self.nembed = nembed
        
        # save hyperparameters for logging 
        self.save_hyperparameters()
        

    def reparameterize(self, z_mu, z_log_var):
        # Sample epsilon from standard normal distribution
        eps = torch.randn(z_mu.size(0), z_mu.size(1), device=self.device)
        
        # note that log(x^2) = 2*log(x); hence divide by 2 to get std_dev
        # i.e., std_dev = exp(log(std_dev^2)/2) = exp(log(var)/2)
        z = z_mu + eps * torch.exp(z_log_var/2.) 
        return z
    
 
    def encoder(self,x):
        x = self.embed(x)
        x = x.permute(0,2,1) # swap length and channel dims

        x = self.enc_conv_1(x)
        x = F.leaky_relu(x)

        x = self.enc_conv_2(x)
        x = F.leaky_relu(x)

        x = x.view(-1,self.nparam) # flatten
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        
        return z_mean, z_log_var, encoded


    def decoder(self, encoded):
        x = self.dec_linear_1(encoded)
        x = x.view(-1,4*self.edim,(self.slen-2*(self.ks-1)))
        
        x = self.dec_deconv_1(x)
        x = F.leaky_relu(x)
        
        x = self.dec_deconv_2(x)
        x = F.leaky_relu(x)
        
        x = x.permute(0,2,1) # swap channel and length dims
        x = self.rev_embed(x)
        decoded = x.permute(0,2,1) # need to permute back
        
        return decoded

    
    def forward(self, x):
        z_mean, z_log_var, encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return z_mean, z_log_var, encoded, decoded

        
    def training_step(self, batch, batch_idx):
        # pass through network 
        z_mean, z_log_var, encoded, decoded = self(batch)

        # cost = reconstruction loss + Kullback-Leibler divergence
        kl_divergence = (0.5 * (z_mean**2 + torch.exp(z_log_var) - z_log_var - 1)).sum()
        ce_loss = F.cross_entropy(decoded,batch,reduction='sum')
        cost = kl_divergence + ce_loss 
        
        # log 
        self.log("train_ce_loss", ce_loss, prog_bar=True, logger=True, on_step = False, on_epoch=True)

        return cost

    
    def validation_step(self, batch, batch_idx):
        # pass through network 
        z_mean, z_log_var, encoded, decoded = self(batch)

        # cost = reconstruction loss + Kullback-Leibler divergence
        kl_divergence = (0.5 * (z_mean**2 + torch.exp(z_log_var) - z_log_var - 1)).sum()
        ce_loss = F.cross_entropy(decoded,batch,reduction='sum')
        cost = kl_divergence + ce_loss 
        
        # log 
        self.log("val_ce_loss", ce_loss, prog_bar=True, logger=True, on_step = False, on_epoch=True)

        return cost      
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class ProtDCA(pl.LightningModule):
        
    def __init__(self, slen, learning_rate=0.0001):
        super().__init__()
      
        self.slen = slen # sequence length
        self.learning_rate = learning_rate
        self.embed = nn.Embedding.from_pretrained(torch.eye(21), freeze=True)
        self.linear = torch.nn.Linear(slen*21,slen*21)
        self.mask = (1-torch.block_diag(*[torch.ones(21,21,device=self.device)]*slen)) # needs to be on GPU (device 0?)
        
        # save hyperparameters for logging 
        self.save_hyperparameters()
        
    
    def forward(self, x):
        x = self.embed(x)
        x = x.view(-1,self.slen*21)
        
        # zero out parameters at same positions
        self.linear._parameters['weight'] =  self.linear._parameters['weight'] * self.mask

        # force symmetric 
        self.linear._parameters['weight'] = 0.5*(self.linear._parameters['weight'] + self.linear._parameters['weight'].T) 
        
        # apply and reshape 
        x = self.linear(x)
        x = x.view(-1,21,self.slen)

        return x
        
        
    def training_step(self, batch, batch_idx):
        weights, batch = batch[1], batch[0]

        # pass through network 
        out = self(batch)
        
        # calc loss 
        ce_loss = (F.cross_entropy(out,batch,reduction='none') * weights[:, None]).sum()
        
        # log 
        self.log("train_ce_loss", ce_loss, prog_bar=True, logger=True, on_step = False, on_epoch=True)

        return ce_loss

    
    def validation_step(self, batch, batch_idx):
        weights, batch = batch[1], batch[0]
        # pass through network 
        out = self(batch)
        
        # calc loss
        ce_loss = (F.cross_entropy(out,batch,reduction='none') * weights[:, None]).sum()
        
        # log 
        self.log("val_ce_loss", ce_loss, prog_bar=True, logger=True, on_step = False, on_epoch=True)

        return ce_loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
