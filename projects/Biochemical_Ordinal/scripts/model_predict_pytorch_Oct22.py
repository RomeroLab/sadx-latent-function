import pathlib

import numpy as np
import pandas as pd

import Bio
import Bio.SeqIO

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

import dataset_Oct22
import model_config_Oct22

from model_pytorch_Oct22 import LightningMLP, BaseCoralModule

import utils
utils.add_projects_to_path()
from protein_utils.sequences.encoding import DEFAULT_NO_GAP_ENCODER
from protein_utils.energy.potts import energy_calc_single_mutants, create_single_mutant


class SingleMutant_Oct22DataSet(Dataset):

    def __init__(self, 
                 encoding="one-hot", target="multiclass", multilibrary=False, 
                 **kwargs):
        self.q = dataset_Oct22.q
        self.multilibrary = multilibrary
        self.wt = utils.get_parent_seq("3VRL")[1:-1]
        self.wt_np = DEFAULT_NO_GAP_ENCODER.string_to_np(self.wt)
        self.L = len(self.wt)

        bmDCA_params_dir = "../data/scratch/bmDCA"
        # these params include the gap character
        e_i_a_j_b = np.load(f"{bmDCA_params_dir}/sadA_e_i_a_j_b.npy")
        h_i_a = np.load(f"{bmDCA_params_dir}/sadA_h_i_a.npy")
        
        # energies of single mutants
        self.sm, self.dca_score = \
                energy_calc_single_mutants(self.wt_np, 
                        h_i_a=h_i_a, e_i_a_j_b=e_i_a_j_b)

        self.no_gap_sm_list = [idx for idx, (i, a) in enumerate(self.sm) if a < 20]

        #self.data[self.data.parent == "1VH"].copy().reset_index()
        self.encoding_func = lambda x: x
        if encoding == "one-hot":
            # one_hot_encode_list takes a list of sequences and one-hot encodes them
            # so we send in a singleton list and squeeze the output
            self.encoding_func = dataset_Oct22.one_hot_encode_single_np
        assert(target in ["multiclass", "binary"])
        self.target = target 
        self.__dict__.update(kwargs)

    def __len__(self):
        return len(self.no_gap_sm_list)

    def __getitem__(self, idx):
        idx = self.no_gap_sm_list[idx] # index from the list of single mutants without a gap
        i, a = self.sm[idx]
        dx = {}
        mut = create_single_mutant(i, a, self.wt_np)
        dx['encoding'] = self.encoding_func(mut).astype(np.float32)
        dx['dca'] = self.dca_score[idx].astype(np.float32)
        if self.multilibrary:
            dx['dataset_num'] = np.int64(2) # third parent
        else:
            raise ValueError("multilibrary must be set to true")
        if self.target == "multiclass":
            target = np.int64(1)
        else:
            raise ValueError("target must be multiclass")
        dx["target"] = target
        return dx

    def get_single_mutants_as_pd(self):
        """ Return dataframe of single mutants """
        # get the no-gap list of single mutants
        rec = self.sm[self.no_gap_sm_list]
        df = pd.DataFrame({'i':rec.i, 'a':rec.a})
        df["wt_aa"] = df.i.map(lambda x: self.wt[x])
        df["pos"] = df.i + 2
        df["aa"] = df.a.map(DEFAULT_NO_GAP_ENCODER.int_to_alpha_dict)
        df["feature"] = df["wt_aa"] + df.pos.astype(str) + df.aa
        return df

if __name__ == "__main__":
    import sys
    import argparse
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--loglevel",
        help="Logging Level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout,
                            level=getattr(logging, args.loglevel))
    
    logging.info(f"Creating dataset")
    ds = SingleMutant_Oct22DataSet(multilibrary=True)
    logging.info("dataset : " + str(ds).replace("\n", ", "))


    saved_model_stub = pathlib.Path(
            "../output/ordinal_Oct22_models/saved_models/6eeae50e")
    mc = model_config_Oct22.ModelConfig.create_from_yaml(
         saved_model_stub.with_suffix(".yml"))
    logging.info("dataset : " + str(mc).replace("\n", ", "))


    dca = True
    additional_features = 0
    if dca: additional_features = 1
    pytorch_model = BaseCoralModule(
        input_size=ds.L*dataset_Oct22.q + additional_features,
        hidden_units=(100, 50),
        num_classes=dataset_Oct22.NUM_CLASSES, 
        num_datasets=dataset_Oct22.NUM_DATASETS)
    model = LightningMLP(
        model=pytorch_model,
        learning_rate=mc.training_params["learning_rate"],
        weight_decay=mc.training_params["weight_decay"],
        dca = mc.design_matrix["dca"])
    #model.xavier_init()

    # FIXME: train on whole dataset
    trained_model = LightningMLP.load_from_checkpoint(checkpoint_path=
            saved_model_stub.with_suffix(".ckpt"), model=pytorch_model)

    trainer = pl.Trainer()

    # get predict data
    predict_dl = DataLoader(ds, batch_size=32,
                            num_workers=4)
    logging.info(f"len(predict_dl) : {len(predict_dl)}")
    probas, true_labels, predicted_labels = list(zip(*trainer.predict(model, predict_dl)))
    probas = torch.cat(probas, dim=0)
    predicted_labels = torch.hstack(predicted_labels)
    print(np.unique(predicted_labels, return_counts=True))
    cum_probs = np.hstack([
                    np.ones((probas.shape[0], 1)), 
                    probas, 
                    np.zeros((probas.shape[0], 1))])
    probs = -np.diff(cum_probs)
    df = ds.get_single_mutants_as_pd()
    df["Hbin_probs"] = probs[:, -1]
    df_sort = df.sort_values(by="Hbin_probs", ascending=False)
    print(df_sort.head())
    print(df_sort.tail())
    df_sort[["feature", "Hbin_probs"]].to_csv(
            saved_model_stub.with_suffix(".Hbin_probs.tsv"), sep="\t", index=False)
    np.savetxt(saved_model_stub.with_suffix(f".probs.txt"), probs)


