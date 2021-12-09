"""Data Modules to load rosetta variant data for pytorch lightning models"""

import numpy as np
import pandas as pd

import torch
import torch.utils.data as data_utils

import pytorch_lightning as pl

# local imports
import rosetta_db
import utils


AAs = 'ACDEFGHIKLMNPQRSTVWY'
aa2ind_dict = {a:i for i, a in enumerate(AAs)}

class RosettaSQLdf(torch.utils.data.Dataset):
   
    def __init__(self, df, parent_seq):
        self.df = df
        self.parent_seq = parent_seq
        self.parent_enc = list(map(aa2ind_dict.get, self.parent_seq))
        
    def __getitem__(self, idx):
        """ All terms are energy terms except the last term (variant)
        """
        energies = torch.tensor(self.df.iloc[idx][:-1], dtype=torch.float32)
        seq_list = utils.expand_mut_str_list_to_list(
                self.df.iloc[idx, -1], self.parent_enc, 
                encoder=aa2ind_dict.get, split_mut_char=".", offset=1)
        seq = torch.tensor(seq_list, dtype=torch.int)
        return seq, energies

    def __len__(self):
        return len(self.df)

class RosettaEnergiesDataModule(pl.LightningDataModule):
    
    def __init__(self, parent = "2D", num_train_variants=-1, batch_size=32, 
                 num_workers=1):
        """ num_train_variants == -1 means use all variants to train """
        super().__init__()
        self.parent = parent
        self.num_train_variants = num_train_variants
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # read sequence and encode
        self.parent_seq = utils.get_parent_seq(parent=self.parent)
        self.slen = len(self.parent_seq)
        #self.parent_enc = list(map(aa2ind_dict.get, self.parent))

        # read in data from training/test split
        self.splits_dir = rosetta_db.get_splits_dir(parent=self.parent)

        self.df = pd.read_pickle(self.splits_dir / "train_df.pkl")
        self.columns_to_drop = utils.get_columns_below_std_threshold(
                df=self.df, std_threshold=0.001)
        self.df.drop(columns=self.columns_to_drop, inplace=True)

        # number of parameters to predict dropping last column (variant)
        self.nparam = len(self.df.columns) - 1 
        self.column_names = list(self.df.columns)[:-1]

        N = len(self.df) # only 90% of the dataset as 10% is the hidden test set
        n_train = int(80/90 * N) # 80% of the total dataset
        n_val = N - n_train # 10% of the total dataset
        self.train_idx, self.val_idx = \
                data_utils.random_split(range(N),[n_train, n_val])

        # Now reduce the number of training variants if requested
        if self.num_train_variants <= 0:
            self.num_train_variants = N
        if self.num_train_variants < N: 
            # num_train_variants passed in was something other than 0 or -1 
            # Also, it is less than N so we should subset train_idx and val_idx
            n_train = int(self.num_train_variants * 80/90)
            self.train_idx = data_utils.Subset(self.train_idx, 
                    torch.arange(n_train))
            self.val_idx = data_utils.Subset(self.val_idx,
                    torch.arange(self.num_train_variants - n_train))


    def setup(self, stage=None):
              
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_data = RosettaSQLdf(
                    self.df.iloc[list(self.train_idx),], self.parent_seq)
            self.val_data = RosettaSQLdf(
                    self.df.iloc[list(self.val_idx),], self.parent_seq)
            
        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            test_df = pd.read_pickle(self.splits_dir / "test_df.pkl")
            test_df.drop(columns=self.columns_to_drop, inplace=True)
            self.test_data = RosettaSQLdf(test_df, self.parent_seq)

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
    print(f"Sequence length                   : {rd.slen}")
    print(f"Total number of params to predict : {rd.nparam}")
    for x in rd.train_dataloader():
        break
    print(f"Batch sequences shape: {x[0].shape}, dtype:{x[0].dtype}")
    print(f"Batch energies shape : {x[1].shape}, dtype:{x[1].dtype}")
