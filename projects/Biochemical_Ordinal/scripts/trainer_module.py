
import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

# local imports
from data_module import RosettaEnergiesDataModule
import model_module

import warnings
warnings.filterwarnings("ignore", 
            ".*Consider increasing the value of the `num_workers` argument*")
warnings.filterwarnings("ignore", 
            ".*Set a lower value for log_every_n_steps "
            "if you want to see logs for the training epoch")


def get_metrics_as_df(metrics_csv_file):
    """ Read metrics csv file from CSVLogger and reshape and return"""
    metrics = pd.read_csv(metrics_csv_file)
    metrics_melt = pd.melt(metrics, id_vars=["epoch", "step"]).dropna()
    return metrics_melt

def get_predictions(trainer, model, dm):
    """ Run predictions on the test set """
    dm.setup("test") # as we predict on the test set

    ret = trainer.predict(model, dm) # ret is in batches
    # Concat tensors in ret to get predictions and values
    predictions = torch.cat([batch[0].sum(dim=1) for batch in ret]).numpy()
    values = torch.cat([batch[1].sum(dim=1) for batch in ret]).numpy()

    return predictions, values



def train(model_name, num_variants=100000, 
            embed_ncomp=0, # one-hot encoding
            embed_freeze=True, # whether the embedding is trainable or not
            max_epochs=50, gpus=0, seed = 1111, 
            log_dir="../data/scratch/rosetta/training_logs",
            **model_kwargs):

    pl.utilities.seed.seed_everything(seed)
    logger = CSVLogger(log_dir, name='RosettaEnergiesTraining')
    dm = RosettaEnergiesDataModule(parent="2D", num_variants=num_variants)
    trainer = pl.Trainer(logger=logger, max_epochs=max_epochs, 
                            gpus=gpus)
    # initialize model from the model_name
    model = getattr(model_module, model_name)(slen=dm.slen,
                        nparam=dm.nparam, embed_ncomp=embed_ncomp, 
                        embed_freeze=embed_freeze, **model_kwargs)
    trainer.fit(model, dm)

    # return info on training process
    retd = {'num_variants':num_variants, 'max_epochs':max_epochs, 'seed':seed,
            'embed_ncomp':embed_ncomp, 'embed_freeze':embed_freeze}
    retd.update(model_kwargs)
    retd['log_dir'] = logger.log_dir #logging.log_dir is subdir of log_dir

    retd["metrics"] = get_metrics_as_df(f"{logger.log_dir}/metrics.csv")
    retd["predictions"], retd["values"] = get_predictions(trainer, model, dm)

    return model, retd

if __name__ == "__main__":
    import pickle
    import pathlib
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed",
                    help="Set global seed", default=1111, type=int) 
    parser.add_argument("-e", "--max_epochs",
                    help="Max number of epochs", default=5, type=int)
    parser.add_argument("-o", "--output_pickle",
                    help="Output pickle file", default=None)
    parser.add_argument("-n", "--num_variants",
                    help="Num variants", default=1000, type=int) 
    parser.add_argument("-m", "--model_name",
                    help="Model name from model_module.py", 
                    default='LinearRosettaEnergyPrediction') 
    parser.add_argument("-c", "--embed_ncomp",
                    help="Dimension of PCA embedding (0 for one-hot)", 
                    default=0, type=int) 
    parser.add_argument("-u", "--embed_unfreeze", # make embedding trainable?
                    help="Whether the embedding is trainable or not", 
                    action='store_true')  # default False i.e. frozen embedding
    parser.add_argument("-g", "--gpus",
                    help="GPUs", default=0, type=int) 
    args = parser.parse_args()

    model, retd = train(
            model_name=args.model_name, 
            num_variants=args.num_variants, 
            embed_ncomp=args.embed_ncomp, # one-hot encoding
            embed_freeze=not args.embed_unfreeze, # embedding is trainable? 
            max_epochs=args.max_epochs, 
            gpus=args.gpus, 
            seed = args.seed) 

    if args.output_pickle is not None:
        op_filename = pathlib.Path(args.output_pickle).with_suffix(".pkl")
        with open(op_filename, 'wb') as op:
            pickle.dump(retd, op)



