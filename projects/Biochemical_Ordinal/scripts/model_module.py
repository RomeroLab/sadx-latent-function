
import torch
import torch.optim 
import torch.nn as nn


import pytorch_lightning as pl

# local imports
from train_vaes_phils_nb import get_aaindex_embedding


class RosettaEnergyPrediction(pl.LightningModule):
    
    def __init__(self, 
                slen,  # sequence length (get from data_module dm.slen)
                nparam, # number of energy terms to predict (dm.nparam)
                embed_ncomp=0,  # 0 means one-hot, >0 means PCA encoding
                embed_freeze=True, # whether the embedding is trainable or not
                learning_rate=0.01):
        
        super().__init__()

        ## GENERAL INPUT PARAMETERS
        self.slen = slen # sequence length
        self.nparam = nparam # number of energy terms to predict
        self.q = 20 # amino acid alphabet_size (without gaps)
        
        self.learning_rate = learning_rate
        
        if embed_ncomp: 
            # if non-zero get a pretrained PCA embedding of Amino acids based
            # on physiochemical properties
            aaindex = get_aaindex_embedding(ncomp=embed_ncomp)
            aaindex = aaindex[:-1, :] # drop the gap character
            assert(aaindex.shape[0] == self.q)
        else: # if embed_ncomp is zero then do one-hot encoding
            aaindex = torch.eye(self.q)
        self.embed = nn.Embedding.from_pretrained(aaindex, freeze=embed_freeze)
        self.edim = self.embed.embedding_dim # dimensions of AA embedding
            
        self.setup_layers()

        # save hyperparameters for logging 
        self.save_hyperparameters()
            

    def setup_layers(self):
        raise NotImplementedError
        
    def forward(self, x):
        raise NotImplementedError
    
    def shared_step(self, batch):
        x, y = batch
        # pass through network
        out = self(x)
        
        # calculate loss
        return nn.MSELoss()(out, y)
        
    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch)
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x), y
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer



class LinearRosettaEnergyPrediction(RosettaEnergyPrediction):

    def setup_layers(self):
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self.edim*self.slen, self.nparam)

    def forward(self, x):
        x = self.embed(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x
  
