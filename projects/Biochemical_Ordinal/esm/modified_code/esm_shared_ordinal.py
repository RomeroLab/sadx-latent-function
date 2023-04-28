""" Add new classes that help with ordinal regression """

from argparse import ArgumentParser
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.utils.data as data_utils

# FIXME remove random_split and do that somewhere else
from torch.utils.data import random_split

import pandas as pd
import pytorch_lightning as pl

from esm_shared import ESMCollate, ESMDataset

class ESMDataModule(pl.LightningDataModule):
    """ Datamodule for loading DMS data encoded for ESM models """

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--ds_fn",
                            help="filename of the csv/hdf5 dataset",
                            type=str, default="data/ordinal_Oct22_sequences_with_dca_score.csv")

        parser.add_argument("--encoding",
                            help="which data encoding to use. should be int_seqs if using an embedding."
                                 " for backwards compat, auto will adopt old behavior of choosing encoding"
                                 " based on the model type",
                            type=str, default="auto")

        parser.add_argument("--batch_size",
                            help="batch size for the data loader and optimizer",
                            type=int, default=32)

        return parser


    def __init__(self, alphabet: "esm.data.Alphabet", 
                 ds_fn: str,
                 batch_size: int = 32,
                 num_dataloader_workers: int = 4,
                 *args, **kwargs):

        super().__init__()
        # ESM alphabet for encoding data
        self.alphabet = alphabet
        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers
        self.ds_fn = ds_fn

        self.example_input_array = self._init_example_input_array(None)
        self.aa_seq_len = len(self.example_input_array['x']['char_seqs'][0])
        
        self.train_name = 'train'
        self.val_name = 'val'
        self.test_name = 'test'

        # initialize DMSDataset that are used later in this module to load dataloaders, etc
        self.full_ds = self.get_ds(None)
        N = len(self.full_ds)
        # FIXME: for testing purposes only. Need to shuffle these and save?
        train_size = int(N * 0.7)
        val_size  = int(N * 0.15)
        self.train_ds, self.val_df, self.test_ds = \
                random_split(self.full_ds, [train_size, val_size, 
                                N - train_size - val_size])
        self.test_ds = self.full_ds[int(N*0.85):]

    def _init_example_input_array(self, sample_batch):
        # create an example input array using actual data
        # will be slower than creating artificial data... but also easier for now :)
        ds = self.get_ds(set_name=None)
        collate = ESMCollate(alphabet=self.alphabet)
        sample_batch = collate([ds[i] for i in range(self.batch_size)])
        return {"x": sample_batch}

    def get_ds(self, set_name: Optional[str]):
        df = pd.read_csv(self.ds_fn)
        variants = "s_" + df.parent + "_" + df.index.astype(str)
        char_seqs = df.sequence_aa_trim
        targets = df.response
        return ESMDataset(sequence_labels=variants,
                          sequence_strs=char_seqs,
                          targets=None if targets is None else torch.from_numpy(targets.to_numpy()))

    def _get_dataloader(self, ds):
        """ helper function for loading train, val, and test dataloaders """
        if ds is None:
            # return None for the dataloader if the underlying dataset is None
            # handles the case for when there is no validation set and val dataloader should be None
            return None
        else:
            return data_utils.DataLoader(ds,
                                         batch_size=self.batch_size,
                                         num_workers=self.num_dataloader_workers,
                                         collate_fn=ESMCollate(self.alphabet),
                                         persistent_workers=True if self.num_dataloader_workers > 0 else False)




    def train_dataloader(self):
        return self._get_dataloader(self.train_ds)

    def val_dataloader(self):
        return self._get_dataloader(self.val_ds)

    def test_dataloader(self):
        return self._get_dataloader(self.test_ds)

    def predict_dataloader(self):
        if self.predict_mode == "all_sets":
            return [self.train_dataloader(), self.val_dataloader(), self.test_dataloader()]
        elif self.predict_mode == "train_set":
            return self.train_dataloader()
        elif self.predict_mode == "full_dataset":
            return self._get_dataloader(self.full_ds)

            # handles the case for when there is no validation set and val dataloader should be None
            return None
        else:
            return data_utils.DataLoader(ds,
                                         batch_size=self.batch_size,
                                         num_workers=self.num_dataloader_workers,
                                         collate_fn=ESMCollate(self.alphabet),
                                         persistent_workers=True if self.num_dataloader_workers > 0 else False)

    def prepare_data(self):
        # prepare_data is called from a single GPU. Do not use it to assign state (self.x = y)
        # use this method to do things that might write to disk or that need to be done only from a single process
        # in distributed settings.
        pass

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_ds = self.get_ds(self.train_name)
            self.val_ds = self.get_ds(self.val_name)

        if stage == 'test' or stage is None:
            self.test_ds = self.get_ds(self.test_name)


