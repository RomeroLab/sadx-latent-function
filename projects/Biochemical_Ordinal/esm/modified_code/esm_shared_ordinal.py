""" Add new classes that help with ordinal regression """

from argparse import ArgumentParser
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torch import Tensor
from torch.utils.data import Dataset

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
                            type=str, default="data/rosetta_data/gb1_sample/gb1_sample.h5")

        parser.add_argument("--encoding",
                            help="which data encoding to use. should be int_seqs if using an embedding."
                                 " for backwards compat, auto will adopt old behavior of choosing encoding"
                                 " based on the model type",
                            type=str, default="auto")

        parser.add_argument("--split_dir",
                            help="the directory containing the train/tune/test split",
                            type=str, default="data/rosetta_data/gb1_sample/splits/standard_tr0.8_tu0.1_te0.1_w3da7a2fd8b08_r11")
        parser.add_argument("--train_name",
                            help="name of the train set in the split dir",
                            type=str, default="train")
        parser.add_argument("--val_name",
                            help="name of the validation set in the split dir",
                            type=str, default="val")
        parser.add_argument("--test_name",
                            help="name of the test set in the split dir",
                            type=str, default="test")

        parser.add_argument("--target_names",
                            help="names of rosetta energies to use as targets (overrides exclude)",
                            type=str, nargs="+", default=None)
        parser.add_argument("--target_names_exclude",
                            help="names of rosetta energies to exclude",
                            type=str, nargs="*", default=['filter_total_score', 'dslf_fa13', 'res_count_all',
                                                          'linear_chainbreak', 'overlap_chainbreak'])

        parser.add_argument("--batch_size",
                            help="batch size for the data loader and optimizer",
                            type=int, default=32)

        return parser


    def __init__(self, alphabet: "esm.data.Alphabet", *args, **kwargs):

        super().__init__()
        # ESM alphabet for encoding data
        self.alphabet = alphabet
        self.aa_seq_len = 272
        self.batch_size = 32
        self.num_dataloader_workers = 4
        self.train_name = "train"
        self.val_name = "val"
        self.test_name = "test"

        self.example_input_array = self._init_example_input_array(None)

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
        df = pd.read_csv("../../output/ordinal_Oct22_sequences_with_dca_score.csv")
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


