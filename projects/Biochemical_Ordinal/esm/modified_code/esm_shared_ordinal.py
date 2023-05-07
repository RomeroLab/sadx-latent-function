""" Add new classes that help with ordinal regression """

from argparse import ArgumentParser
from typing import Optional, Sequence, Union, Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset
import torch.utils.data as data_utils

import pandas as pd
import pytorch_lightning as pl

import dataset_Oct22



class ESMOrdinalDataset(Dataset):
    """ This is for compatability with the ESM alphabet and batch_converter collate_fn
        Allows us to optionally incorporate DMS target scores """

    def __init__(self,
                 sequence_labels: Union[list[str], tuple[str]],
                 sequence_strs: Union[list[str], tuple[str]],
                 dataset_nums: Optional[Tensor],
                 targets: Optional[Tensor]):

        self.sequence_labels = list(sequence_labels)
        self.sequence_strs = list(sequence_strs)
        self.dataset_nums = dataset_nums
        self.targets = targets

    def __len__(self):
        return len(self.sequence_labels)

    def __getitem__(self, idx):
        out = {"variants": self.sequence_labels[idx],
               "char_seqs": self.sequence_strs[idx]}

        if self.targets is None:
            # handle the case where we don't have DMS targets
            out["targets"] = None
        else:
            out["targets"] = self.targets[idx]

        if self.dataset_nums is None:
            out["dataset_nums"] = None
        else:
            out["dataset_nums"] = self.dataset_nums[idx]

        return out

class ESMCollate:
    """ Returns a collate_fn that wraps ESM's batch_converter and additionally supports DMS targets """
    def __init__(self, alphabet: "esm.data.Alphabet"):
        self.alphabet = alphabet

    def __call__(self, batch):
        # batch is expected to be a dictionary of (variants, char_seqs, targets)
        # targets can optionally be None if the dataset is being loaded without dms targets
        # for example, if only doing inference, dms targets are optional...

        variants = [d["variants"] for d in batch]
        char_seqs = [d["char_seqs"] for d in batch]
        targets = [d["targets"] for d in batch]
        dataset_nums = [d["dataset_nums"] for d in batch]

        # ESM batch_converter takes a sequence of tuples of variants and char_seqs
        # and returns a tuple of (variants, char_seqs, encoded_data)
        bc = self.alphabet.get_batch_converter()
        variants, char_seqs, encoded_data = bc(list(zip(variants, char_seqs)))

        # collate targets using the default collate
        # default_collate does not support None types
        # so first check for None and collate ourselves by collapsing down to single None value
        if all(v is None for v in targets):
            targets = None
        elif any(v is None for v in targets):
            raise ValueError("only some targets are 'None'...this shouldn't happen")
        else:
            targets = torch.utils.data.default_collate(targets)
            dataset_nums = torch.utils.data.default_collate(dataset_nums)

        return {"variants": variants,
                "char_seqs": char_seqs,
                "encoded_data": encoded_data,
                "targets": targets,
                "dataset_nums": dataset_nums}



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
                 predict_mode: Literal["all_sets", "train_set", "full_dataset"] = "all_sets",
                 num_dataloader_workers: int = 4,
                 *args, **kwargs):

        super().__init__()
        # ESM alphabet for encoding data
        self.alphabet = alphabet
        self.batch_size = batch_size
        self.predict_mode = predict_mode
        self.num_dataloader_workers = num_dataloader_workers
        self.ds = dataset_Oct22.Oct22DataSet(ds_fn)

        self.has_val_set = True
        
        self.train_name = 'train'
        self.val_name = 'val'
        self.test_name = 'test'

        # initialize DMSDataset that are used later in this module to load dataloaders, etc
        self.test_df = self.ds.get_test_dataset()
        # split train into train and val using cv1
        for self.train_df, self.val_df in self.ds.cv_iterator():
            break

        self.example_input_array = self._init_example_input_array(None)
        self.aa_seq_len = len(self.example_input_array['x']['char_seqs'][0])

    def _init_example_input_array(self, sample_batch):
        # create an example input array using actual data
        # will be slower than creating artificial data... but also easier for now :)
        ds = self.get_ds(set_name=None)
        collate = ESMCollate(alphabet=self.alphabet)
        sample_batch = collate([ds[i] for i in range(self.batch_size)])
        return {"x": sample_batch}

    def get_ds(self, set_name: Optional[str]):
        df = self.train_df
        if set_name == "test":
            df = self.test_df
        elif set_name == "val":
            df = self.val_df
        variants = self.get_variants(set_name)

        char_seqs = df.sequence_aa_trim
        dataset_nums = torch.from_numpy(df.dataset_num.to_numpy())
        #targets = df.response
        targets = self.get_targets(set_name)
        return ESMOrdinalDataset(sequence_labels=variants,
                          sequence_strs=char_seqs,
                          dataset_nums = dataset_nums,
                          targets=None if targets is None else torch.from_numpy(targets.to_numpy()).float())

    def get_targets(self, set_name, *args, **kwargs):
        df = self.train_df
        if set_name == "test":
            df = self.test_df
        elif set_name == "val":
            df = self.val_df
        return df.response


    def get_variants(self, set_name, *args, **kwargs):
        df = self.train_df
        if set_name == "test":
            df = self.test_df
        elif set_name == "val":
            df = self.val_df
        return "s_" + df.parent + "_" + df.index.astype(str)


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


