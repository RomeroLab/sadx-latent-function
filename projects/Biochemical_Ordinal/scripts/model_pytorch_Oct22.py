import pathlib

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from coral_multiple_layer import CoralMultipleLayer
from coral_pytorch.losses import coral_loss
from coral_pytorch.dataset import levels_from_labelbatch, proba_to_label

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
    group.add_argument("--num_epochs", help="Num Epochs", default=50, type=int)
    group.add_argument("--num_workers", help="Num workers", default=4, type=int)
    group.add_argument("--lambda_h", help="Regularization param", default=1e-4, type=float)
    group.add_argument("--lambda_dca", help="DCA Regularization param", default=1e-8, type=float)
    group.add_argument("--learning_rate", help="Learning Rate", default=1e-4, type=float)
    group.add_argument("--weight_decay", 
                        help="Weight decay (conflicts with lambda_h)", 
                        default=1e-6 , type=float)
    return group

class Pytorch_Oct22DataSet(Dataset):

    @classmethod
    def create_from_args(cls, args, dataset_name):
        return cls(
                data_csv_fn = args.input,
                dataset_name = dataset_name,
                target = args.target,
                encoding = args.encoding,
                multilibrary = args.multilibrary
                )
   
    def __init__(self, data_csv_fn, dataset_name="train", 
                 encoding="one-hot", target="multiclass", multilibrary=False, 
                 **kwargs):
        ds = dataset_Oct22.Oct22DataSet(data_csv_fn=data_csv_fn)
        self.L = ds.L
        self.q = dataset_Oct22.q
        self.multilibrary = multilibrary
        self.data = ds.get_dataset_by_name(dataset_name).copy().reset_index()
        self.encoding_func = lambda x: x
        if encoding == "one-hot":
            # one_hot_encode_list takes a list of sequences and one-hot encodes them
            # so we send in a singleton list and squeeze the output
            self.encoding_func = dataset_Oct22.one_hot_encode_single
        assert(target in ["multiclass", "binary"])
        self.target = target
        self.__dict__.update(kwargs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dx = {}
        data_row = self.data.iloc[idx]
        dx['encoding'] = self.encoding_func(data_row.sequence_aa_trim).astype(np.float32)
        dx['dca'] = data_row.dca_score.astype(np.float32)
        if self.multilibrary:
            dx['dataset_num'] = np.int64(data_row.dataset_num)
        else:
            dx['dataset_num'] = np.int64(0)
        if self.target == "multiclass":
            target = data_row.activity_num.astype(np.int64)
        else:
            target = (data_row.category_num != 0).astype(np.int64)
        dx["target"] = target
        return dx


class PytorchTrainingTask:

    @classmethod
    def create_from_args(cls, args):
        return cls()

    def __init__(self, batch_size=32, num_workers=4, num_epochs=50):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs

class PytorchRegression(pl.LightningModule):

    @classmethod
    def create_from_args(cls, args):
        raise NotImplementedError
        return cls(
                )
 
    def __init__(self, 
            lambda_h = 1e-4, lambda_dca = 1e-8,
            learning_rate = 1e-4,
            weight_decay = 1e-6,
            random_state=100, 
            L = 272, q = dataset_Oct22.q, encoding = "one-hot", 
            target = "multiclass", dca=False, multilibrary = False,
            **kwargs):
        super().__init__()

        self.lambda_h = lambda_h
        self.lambda_dca = lambda_dca
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.L = L
        self.q = q
        self.encoding = encoding
        self.target = target
        self.dca = dca
        self.multilibrary = multilibrary
        self.num_classes = dataset_Oct22.NUM_CLASSES
        self.num_datasets = dataset_Oct22.NUM_DATASETS

        size_in = 0
        if encoding == "one-hot":
            size_in = self.L*self.q 
        else:
            raise NotImplementedError("Only one-hot encoding for now")

        additional_features = 0
        if self.dca:
            additional_features += 1
        size_in += additional_features

        self.model = self._build_layers(None, None)

        if not self.multilibrary:
            self.num_datasets = 1

        self.output_layer = None
        if self.target == "multiclass":
            self.output_layer = CoralMultipleLayer(
                        size_in=size_in,
                        num_classes=self.num_classes, 
                        num_datasets=self.num_datasets)
        elif self.target == "binary":
            self.num_classes = 2
            self.output_layer = torch.nn.Linear(size_in, 1)

 
    def forward(self, x, dx):
        x = self.model(x)
        if self.target == "multiclass":
            x = self.output_layer(x, dx)
        else: # target is binary
            x = self.output_layer(x)
        return x

    def _build_layers(self, input_size, hidden_units):
        """
            Initialize MLP layers. Override this function to make a custom NN 
        """
        return torch.nn.Identity()

    def shared_step(self, batch, batch_idx):
        x, dx, true_labels = batch["encoding"], batch["dataset_num"], batch["target"]
        if self.dca: # add dca as the last value
            x = torch.hstack((x, batch["dca"].unsqueeze(1)))
        logits = self(x, dx)
        loss = None
        if self.target == "multiclass":
            # Convert class labels for CORAL ------------------------
            levels = levels_from_labelbatch(
                true_labels, num_classes=self.num_classes)
            # -------------------------------------------------------

            # CORAL Loss --------------------------------------------
            # A regular classifier uses:
            loss = coral_loss(logits, levels.type_as(logits))
            # -------------------------------------------------------
        else:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.squeeze(), 
                    true_labels.squeeze().type_as(logits))

        # add regularization
        if self.lambda_h:
            last_weights = None
            last_bias = None
            if self.target == "multiclass":
                last_weights = self.output_layer.coral_weights.weight
                last_bias = self.output_layer.coral_bias
            else: #target is binary and the output layer is linear
                last_weights = self.output_layer.weight
                last_bias = self.output_layer.bias
            dca_param = None
            if self.dca:
                last_weights = last_weights[:, :-1]
                # dca score is the last param
                dca_param = last_weights[:, -1]
            loss += self.lambda_h * ((last_weights * last_weights).sum()
                                      + (last_bias * last_bias).sum())
            if self.dca:
                loss += self.lambda_dca * (dca_param * dca_param).sum()
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("validation_loss", loss, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("test_loss", loss, on_epoch=True)
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, dx, true_labels = batch["encoding"], batch["dataset_num"], batch["target"]
        if self.dca: # add dca as the last value
            x = torch.hstack((x, batch["dca"].unsqueeze(1)))
        logits = self(x, dx)
        return logits # return the logits

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, 
                weight_decay=self.weight_decay)

    def fit(self, *args, **kwargs):
        pass


def create_model(model_config):
    """ Takes a model config object and returns a model that can run.
        
        Note: Unlike sklearn models, Pytorch models do not do automatic CV
    """
    mc = model_config
    model = None
    if model_config.model_name == "Pytorch":
        model = PytorchRegression(
                    lambda_h = mc.model_params["lambda_h"],
                    lambda_dca = mc.model_params["lambda_dca"],
                    learning_rate = mc.training_params["learning_rate"],
                    weight_decay = mc.training_params["weight_decay"],
                    random_state = mc.seed,
                    dca = mc.design_matrix["dca"], 
                    target = mc.target,
                    encoding = mc.design_matrix["encoding"],
                    multilibrary = mc.design_matrix["multilibrary"])
    else:
        raise ValueError(f"{model_name} must be one of {model_names}")
    return model



class MetricTracker(pl.Callback):

    def __init__(self):
        self.train_losses = []
        self.validation_losses = []

    def on_train_epoch_end(self, trainer, module):
        elogs = trainer.logged_metrics # access it here
        self.train_losses.append(elogs["train_loss_epoch"].item())

    def on_validation_epoch_end(self, trainer, module):
        elogs = trainer.logged_metrics # access it here
        self.validation_losses.append(elogs["validation_loss_epoch"].item())

    def get_df(self):
        df = pd.DataFrame({'train_losses': np.array(self.train_losses)})
        df['epoch'] = np.arange(len(df)) + 1
        if len(self.train_losses) == len(self.validation_losses) - 1:
            df_val = pd.DataFrame(
                    {"validation_losses": np.array(self.validation_losses)})
            df_val["epoch"] = np.arange(len(df_val))
            df = pd.merge(df_val, df, how="outer")
        return df

    def save_losses_plot(self, filename):
        df = self.get_df()
        df.plot.line(y=[c for c in df.columns if c.endswith("_losses")], x="epoch",
                        figsize=(10,6)).get_figure().savefig(filename)

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

    model = create_model(model_config = mc)

    # get the training data
    train_ds = Pytorch_Oct22DataSet.create_from_args(
                args=args, dataset_name=args.train_name)
    logging.info(f"len(train_ds) : {len(train_ds)}")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, 
                            num_workers=args.num_workers)
    for x in train_dl: break
    logging.info(f"Batch Encoding Shape : {x['encoding'].shape}")

    # get the validation data
    val_dl = None
    if args.val_name:
        val_ds = Pytorch_Oct22DataSet.create_from_args(
                    args=args, dataset_name=args.val_name)
        logging.info(f"len(train_ds) : {len(train_ds)}")

        val_dl = DataLoader(val_ds, batch_size=args.batch_size, 
                                num_workers=args.num_workers)

    # train the model
    cb = MetricTracker()
    trainer = pl.Trainer(max_epochs=args.num_epochs, log_every_n_steps=10, callbacks=[cb])
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    
    cb.save_losses_plot(args.output_dir / f"{mc.uuid}.losses.png")


    # get testing data
    test_ds = Pytorch_Oct22DataSet.create_from_args(
                args=args, dataset_name=args.test_name)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, 
                            num_workers=args.num_workers)
    logging.info(f"len(test_ds) : {len(test_ds)}")
    scores = trainer.predict(model, test_dl)
    scores = torch.vstack(scores)

    probas = torch.sigmoid(scores)
    predicted_labels = proba_to_label(probas)

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
