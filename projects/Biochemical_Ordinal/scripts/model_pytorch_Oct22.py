import pathlib

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

import dataset_Oct22
import model_config_Oct22

from model_shared_Oct22 import \
        add_common_arguments, \
        add_design_arguments, \
        add_data_arguments

model_names = ["Pytorch"]

def add_pytorch_arguments(parser):
    group = parser.add_argument_group("pytorch")
    group.add_argument("--batch_size", help="Batch Size", default=32, type=int)
    group.add_argument("--num_workers", help="Num workers", default=4, type=int)
    group.add_argument("--lambda", help="Regularization param", default=1e-8, type=float)
    return group

class Pytorch_Oct22DataSet(Dataset):

    @classmethod
    def create_from_args(cls, args, dataset_name):
        return cls(
                data_csv_fn = args.input,
                dataset_name = dataset_name,
                target = args.target,
                encoding = args.encoding,
                )
   
    def __init__(self, data_csv_fn, dataset_name="train", 
                 encoding="one-hot", target="multiclass"):
        ds = dataset_Oct22.Oct22DataSet(data_csv_fn=data_csv_fn)
        self.data = ds.get_dataset_by_name(dataset_name).copy().reset_index()
        self.encoding_func = lambda x: x
        if encoding == "one-hot":
            # one_hot_encode_list takes a list of sequences and one-hot encodes them
            # so we send in a singleton list and squeeze the output
            self.encoding_func = dataset_Oct22.one_hot_encode_single
        assert(target in ["multiclass", "binary"])
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dx = {}
        data_row = self.data.iloc[idx]
        dx['encoding'] = self.encoding_func(data_row.sequence_aa_trim)
        dx['dca'] = data_row.dca_score.item()
        dx['dataset_num'] = data_row.dataset_num.item()
        if self.target == "multiclass":
            target = data_row.activity_num.item()
        else:
            target = data_row.category_num.item()
        dx["target"] = target
        return dx

class PytorchOrdinalRegression:

    def __init__(self, *args, **kwargs):
        pass

    def fit(self, *args, **kwargs):
        pass


def create_model(model_config):
    """ Takes a model config object and returns a model that can run.
        
        Note: Unlike sklearn models, Pytorch models do not do automatic CV
    """
    model = None
    if model_config.model_name == "Pytorch":
        model = PytorchOrdinalRegression(
                    random_state = model_config.seed,
                    )
    else:
        raise ValueError(f"{model_name} must be one of {model_names}")
    return model


if __name__ == "__main__":
    import sys
    import argparse
    import logging

    parser = argparse.ArgumentParser()
    group = add_common_arguments(parser, model_names = model_names)
    group = add_design_arguments(parser)
    group = add_pytorch_arguments(parser)
    group = add_data_arguments(parser)
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout,
                            level=getattr(logging, args.loglevel))
    
    logging.info(f"Reading filename: {args.input}")
    ds = dataset_Oct22.Oct22DataSet(args.input)
    logging.info("dataset : " + str(ds).replace("\n", ", "))

    mc = model_config_Oct22.ModelConfig.create_from_args(args)

    logging.info("model_config : " + str(mc).replace("\n", ", "))
    mc.save_yaml(directory=args.output_dir)

    # the dataset object can be used as cv as it iterates over the 
    # split training indices
    model = create_model(model_config = mc)

    # get the training data
    train_ds = Pytorch_Oct22DataSet.create_from_args(
                args=args, dataset_name=args.train_name)
    logging.info(f"len(train_ds) : {len(train_ds)}")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, 
                            num_workers=args.num_workers)
    for x in train_dl: break
    logging.info(f"Batch Encoding Shape : {x['encoding'].shape}")

    # train the model
    #clf = model.fit(X_train, y_train)

    # get testing data
    test_ds = Pytorch_Oct22DataSet.create_from_args(
                args=args, dataset_name=args.test_name)
    logging.info(f"len(test_ds) : {len(test_ds)}")

    ## convert decision scores to probabilities
    ## we do this to look at AUC curves for the binary targets
    #des = model.decision_function(X_test)

    #probs = None
    #argmax_axis = 0
    #if mc.target == "binary":
    #    assert(len(des.shape) == 1)
    #    p = expit(des) # probablity of predicting 1.
    #    probs = np.vstack([1-p, p])
    #    argmax_axis = 0
    #elif mc.target == "multiclass":
    #    assert(des.shape[1] == 4)
    #    probs = softmax(des)
    #    argmax_axis = 1
    #else:
    #    raise ValueError(f"Got unknown value of mc.target={mc.target}")
    #
    #logging.info(f"Saving probabilities")
    #np.savetxt(args.output_dir / f"{mc.uuid}.probs.txt", probs)

    #model_test_predictions = model.predict(X_test)
    #np.savetxt(args.output_dir / f"{mc.uuid}.preds.txt", model_test_predictions)
    #if len(np.unique(model_test_predictions)) == 1:
    #    logging.warning(f"~~ ALERT! all identical predictions on test set !! ~~")

    ## check that the maximum probability is the same as a predict
    #assert((probs.argmax(axis=argmax_axis) == model_test_predictions).all())
    # 
