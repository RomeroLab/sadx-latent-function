""" ESM code shared between esm_inference and esm_model """
from argparse import ArgumentParser
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torch.utils.data import random_split
from torch import Tensor
from torch.utils.data import Dataset

# added by Sameer
import pandas as pd
import pytorch_lightning as pl

import datamodules
import encode as enc


class ESMDataset(Dataset):
    """ This is for compatability with the ESM alphabet and batch_converter collate_fn
        Allows us to optionally incorporate DMS target scores """

    def __init__(self,
                 sequence_labels: Union[list[str], tuple[str]],
                 sequence_strs: Union[list[str], tuple[str]],
                 targets: Optional[Tensor]):

        self.sequence_labels = list(sequence_labels)
        self.sequence_strs = list(sequence_strs)
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

        return {"variants": variants,
                "char_seqs": char_seqs,
                "encoded_data": encoded_data,
                "targets": targets}


class ESMDataModule(datamodules.DMSDataModule):
    """ Datamodule for loading DMS data encoded for ESM models """

    def __init__(self, alphabet: "esm.data.Alphabet", *args, **kwargs):

        # ESM alphabet for encoding data
        self.alphabet = alphabet

        super().__init__(*args, **kwargs)

    def _init_example_input_array(self, sample_batch):
        # create an example input array using actual data
        # will be slower than creating artificial data... but also easier for now :)
        ds = self.get_ds(set_name=None)
        collate = ESMCollate(alphabet=self.alphabet)
        sample_batch = collate([ds[i] for i in range(self.batch_size)])
        return {"x": sample_batch}

    def get_ds(self, set_name: Optional[str]):
        variants = self.get_variants(set_name)
        char_seqs = enc.encode(encoding="char_seqs", variants=variants, ds_name=self.ds_name)
        targets = self.get_targets(set_name)
        return ESMDataset(sequence_labels=variants,
                          sequence_strs=char_seqs,
                          targets=None if targets is None else torch.from_numpy(targets))

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


class ESMSequenceRep(nn.Module):
    """ an ESM wrapper that extracts the sequence-level representation at the top layer
        has an optional seq_len argument if all sequences are the same length (faster)
        note: only supports extracting representations for 1 layer
        for more complex ESM usage, use their extract.py script or add to this implementation """

    def __init__(self,
                 esm_base_model: nn.Module,
                 repr_layer: int,
                 return_contacts: bool = False,
                 seq_len: Optional[int] = None):

        super().__init__()

        self.esm_base = esm_base_model
        self.repr_layer = repr_layer
        self.return_contacts = return_contacts
        self.seq_len = seq_len

    def forward(self, x):
        base_output = self.esm_base(x["encoded_data"],
                                    repr_layers=[self.repr_layer],
                                    return_contacts=self.return_contacts)

        token_reps = base_output["representations"][self.repr_layer]

        if self.seq_len is None:
            # use the length of char_seqs to determine the sequence lengths, which might be variable
            sequence_reps = [token_reps[i, 1: len(x["char_seqs"][i]) + 1].mean(0) for i in range(token_reps.shape[0])]
            sequence_reps = torch.stack(sequence_reps)
        else:
            # assumes that all the sequences are the same length, specified in module init
            # this is the case when we are running on a single DMS dataset
            # this *should* be faster...
            sequence_reps = torch.mean(token_reps[:, 1:(self.seq_len + 1)], dim=1)

        return sequence_reps


### Edit by SAMEER
### Adding classes to read in ordinal data

class ESMOrdDataModule(pl.LightningDataModule):
    """ Datamodule for loading DMS data encoded for ESM models """

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


