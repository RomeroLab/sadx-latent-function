"""Data Modules to load rosetta variant data for pytorch lightning models"""

import numpy as np
import pandas as pd

import torch
import torch.utils.data as data_utils

import pytorch_lightning as pl

# local imports
import utils
from rosetta_db import get_sqlite_dbcon


AAs = 'ACDEFGHIKLMNPQRSTVWY'
aa2ind_dict = {a:i for i, a in enumerate(AAs)}

class RosettaSQLdf(torch.utils.data.Dataset):
   
    def __init__(self, df, parent_seq, dtype=torch.float32):
        self.df = df
        self.dtype = dtype
        self.parent_seq = parent_seq
        self.parent_enc = list(map(aa2ind_dict.get, self.parent_seq))
        
    def __getitem__(self, idx):
        energies = torch.tensor(self.df.iloc[idx][1:-1], dtype=self.dtype)
        seq_list = utils.expand_mut_str_list_to_list(
                self.df.iloc[idx, -1], self.parent_enc, 
                encoder=aa2ind_dict.get, split_mut_char=".", offset=1)
        seq = torch.tensor(seq_list, dtype=torch.int)
        return seq, energies

    def __len__(self):
        return len(self.df)

class RosettaEnergiesDataModule(pl.LightningDataModule):
    
    def __init__(self, parent = "2D", num_variants=100000, batch_size=32, 
                 dtype=torch.float32, num_workers=1):
        super().__init__()
        self.parent = parent
        self.num_variants = num_variants
        self.batch_size = batch_size
        self.dtype = dtype
        self.num_workers = num_workers
        
        # read sequence and encode
        self.parent_seq = utils.get_parent_seq(parent=self.parent)
        self.slen = len(self.parent_seq)
        #self.parent_enc = list(map(aa2ind_dict.get, self.parent))

        # read in data from database
        db_con = get_sqlite_dbcon(parent=self.parent)
        self.df = pd.read_sql(f"select * from relaxed_scores limit "
                              f"{self.num_variants}", db_con)
        
        del self.df["description"]
        del self.df["dslf_fa13"] # disulfide bond energy
        
        # number of parameters to predict
	# dropping first (total energy) and last column (variant)
        self.nparam = len(self.df.columns) - 2 

        N = len(self.df)
        n_train = int(0.8 * N)
        n_val = int(0.1 * N)
        n_test = N - n_train - n_val
        self.train_idx, self.val_idx, self.test_idx = \
                data_utils.random_split(range(N),[n_train, n_val, n_test])

    def setup(self, stage=None):
              
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_data = RosettaSQLdf(
                    self.df.iloc[list(self.train_idx),], self.parent_seq)
            self.val_data = RosettaSQLdf(
                    self.df.iloc[list(self.val_idx),], self.parent_seq)
            
        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_data = RosettaSQLdf(
                    self.df.iloc[list(self.test_idx),], self.parent_seq)

    def train_dataloader(self):
        return data_utils.DataLoader(self.train_data, 
                batch_size=self.batch_size, shuffle=True, 
                num_workers=self.num_workers)

    def val_dataloader(self):
        return data_utils.DataLoader(self.val_data, 
                batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        # make sure shuffle is off here
        return data_utils.DataLoader(self.test_data, 
                batch_size=self.batch_size, num_workers=self.num_workers)
    
    def predict_dataloader(self):
        # just use the test dataloader 
        return self.test_dataloader()


if __name__ == "__main__":
    rd = RosettaEnergiesDataModule(parent="2D")
    rd.setup(stage="fit")
    for x in rd.train_dataloader():
        break
    print(f"Batch sequences shape: {x[0].shape}, dtype:{x[0].dtype}")
    print(f"Batch energies shape : {x[1].shape}, dtype:{x[1].dtype}")
