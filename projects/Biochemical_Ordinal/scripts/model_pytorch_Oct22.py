import pathlib

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from coral_multiple_layer import CoralMultipleLayer
from coral_pytorch.losses import coral_loss
from coral_pytorch.dataset import levels_from_labelbatch, proba_to_label

from multi_coral_pytorch import DataModule, load_synthetic_data, MyDataset

import dataset_Oct22
import model_config_Oct22

from model_shared_Oct22 import \
        add_common_arguments, \
        add_design_arguments, \
        add_data_arguments

model_names = ["MLPLightning", "PytorchRegression"]

def add_pytorch_arguments(parser):
    group = parser.add_argument_group("pytorch")
    group.add_argument("--batch_size", help="Batch Size", default=32, type=int)
    group.add_argument("--num_epochs", help="Num Epochs", default=50, type=int)
    group.add_argument("--num_workers", help="Num workers", default=4, type=int)
    group.add_argument("--lambda_h", help="Regularization param", default=1e-4, type=float)
    group.add_argument("--lambda_dca", help="DCA Regularization param", default=1e-8, type=float)
    group.add_argument("--learning_rate", help="Learning Rate", default=1e-4, type=float)
    group.add_argument("--save_model", help="Save checkpoint", action="store_true")
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
        #self.data[self.data.parent == "1VH"].copy().reset_index()
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
            target = data_row.category_num.astype(np.int64)
        else:
            target = data_row.activity_num.astype(np.int64)
        dx["target"] = target
        return dx


class SyntheticDataset(Dataset):

    def __init__(self, ds):
        self.ds = ds # take a MyDataset and return data in a format like Pytorch_Oct22DataSet
    
    def __getitem__(self, index):
        inputs, dataset, label  = self.ds[index]
        dx = {}
        dx["encoding"] = inputs
        dx["dataset_num"] = dataset
        dx["target"] = label
        return dx


        return inputs, dataset, label

    def __len__(self):
        return len(self.ds)


class Oct22SyntheticDataModule(DataModule):

    def __init__(self, size=1000, round_nums = (1,2,3), 
                 encoding="one-hot", target="multiclass", multilibrary=False, 
                        batch_size=None, num_workers=None ):
        msas, labels, dataset_labels = load_synthetic_data(size, round_nums)
        self.L = msas.shape[1]
        self.q = dataset_Oct22.q
        self.multilibrary = multilibrary
        self.target = target
        assert(self.target == "multiclass")
        kwargs = {}
        if batch_size is not None: kwargs['batch_size'] = batch_size
        if num_workers is not None: kwargs['num_workers'] = batch_size
        super().__init__(syn_data=msas, syn_labels=labels, 
                syn_datasets=dataset_labels, **kwargs)

    def setup(self, stage=None):
        super().setup(stage)
        # now replace MyDataset with a Dataset 
        self.train = SyntheticDataset(self.train)
        self.valid = SyntheticDataset(self.valid)
        self.test = SyntheticDataset(self.test)

    def predict_dataloader(self):
        return self.test_dataloader()


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
                        num_datasets=self.num_datasets, preinit_bias=True)
        elif self.target == "binary":
            self.num_classes = 2
            self.output_layer = torch.nn.Linear(size_in, 1)

        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        self.test_mae = torchmetrics.MeanAbsoluteError()
  
    def forward(self, x, dx):
        #x = self.model(x) # FIXME
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

    def xavier_init(self):
        for name, param in self.named_parameters():
            print (name, param.shape)
            if name.endswith(".bias"): 
                # does not include coral_bias as it does not end with .bias
                # coral_bias is initialized by itself
                param.data.fill_(0)
            else:
                bound = (np.sqrt(6) / np.sqrt(param.shape[0] + param.shape[1])).item()
                param.data.uniform_(-bound, bound)

    def shared_step(self, batch, batch_idx):
        x, dx, true_labels = batch["encoding"], batch["dataset_num"], batch["target"]
        if self.dca: # add dca as the last value
            x = torch.hstack((x, batch["dca"].unsqueeze(1)))
        logits = self(x, dx)
        loss = None
        probas = None
        predicted_labels = None
        if self.target == "multiclass":
            # Convert class labels for CORAL ------------------------
            levels = levels_from_labelbatch(
                true_labels, num_classes=self.num_classes)
            # -------------------------------------------------------

            # CORAL Loss --------------------------------------------
            # A regular classifier uses:
            loss = coral_loss(logits, levels.type_as(logits))
            # -------------------------------------------------------

            # CORAL Prediction to label -----------------------------
            # A regular classifier uses:
            # predicted_labels = torch.argmax(logits, dim=1)
            probas = torch.sigmoid(logits)
            predicted_labels = proba_to_label(probas)
            # -------------------------------------------------------

        else:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.squeeze(), 
                    true_labels.squeeze().type_as(logits))
            probas = torch.sigmoid(logits)
            predicted_labels = (probas > 0.5).squeeze().long()
        # add regularization
        if self.lambda_h:
            print("Doing some regularization")
            last_weights = None
            last_bias = None

            # get the weights and biases from the last layer
            if self.target == "multiclass":
                last_weights = self.output_layer.coral_weights.weight
                last_bias = self.output_layer.coral_bias
            else: #target is binary and the output layer is linear
                last_weights = self.output_layer.weight
                last_bias = self.output_layer.bias

            # Separate out the dca parameter if there is one
            dca_param = None
            if self.dca:
                last_weights = last_weights[:, :-1]
                # dca score is the last param
                dca_param = last_weights[:, -1]
            loss += self.lambda_h * ((last_weights * last_weights).sum()
                                      + (last_bias * last_bias).sum())
            if self.dca and self.lambda_dca:
                loss += self.lambda_dca * (dca_param * dca_param).sum()
        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self.shared_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.train_mae(predicted_labels, true_labels)
        self.log("train_mae", self.train_mae, on_epoch=True, on_step=False)
        #print(loss, true_labels, predicted_labels)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self.shared_step(batch, batch_idx)
        self.log("validation_loss", loss, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self.shared_step(batch, batch_idx)
        self.log("test_loss", loss, on_epoch=True)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        #loss, true_labels, predicted_labels = self.shared_step(batch, batch_idx)
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


# Regular PyTorch Module
class BaseCoralModule(torch.nn.Module):

    def __init__(self, input_size, hidden_units, num_classes, num_datasets):

        super().__init__()

        # num_classes is used by the CORAL loss function
        self.num_classes = num_classes
        self.num_datasets = num_datasets
        
        all_layers = self._build_layers(input_size, hidden_units)

        # CORAL: output layer -------------------------------------------
        # Regular classifier would use the following output layer:
        # output_layer = torch.nn.Linear(hidden_units[-1], num_classes)
        
        # We replace it by the CORAL layer:
        self.output_layer = CoralMultipleLayer(size_in=hidden_units[-1],
                                  num_classes=num_classes, 
                                  num_datasets=num_datasets)
        # ----------------------------------------------------------------
        
        # do not add outputlayer to all layers because
        # we send extra data (dataset information) to it in the forward
        # function
        #all_layers.append(self.output_layer)
        self.model = torch.nn.Sequential(*all_layers)

    def _build_layers(self, input_size, hidden_units):
        """
            Initialize MLP layers. Override this function to make a custom NN 
        """
        all_layers = []
        for hidden_unit in hidden_units:
            layer = torch.nn.Linear(input_size, hidden_unit)
            #all_layers.append(torch.nn.Dropout(0.2))
            all_layers.append(layer)
            all_layers.append(torch.nn.ReLU())
            input_size = hidden_unit
        return all_layers

    def forward(self, x, dx):
        x = self.model(x)
        x = self.output_layer(x, dx)
        return x

    def predict(self, x, dx):
        pass


# LightningModule that receives a PyTorch model as input
class LightningMLP(pl.LightningModule):
    def __init__(self, model, learning_rate, weight_decay, dca=False):
        super().__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        # The inherited PyTorch module
        self.model = model
        self.dca = dca

        # Save settings and hyperparameters to the log directory
        # but skip the model parameters
        self.save_hyperparameters(ignore=['model'])

        # Set up attributes for computing the MAE
        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        self.test_mae = torchmetrics.MeanAbsoluteError()

    def xavier_init(self):
        for name, param in self.named_parameters():
            print (name, param.shape)
            if name.endswith(".bias"): 
                # does not include coral_bias as it does not end with .bias
                # coral_bias is initialized by itself
                param.data.fill_(0)
            else:
                bound = (np.sqrt(6) / np.sqrt(param.shape[0] + param.shape[1])).item()
                param.data.uniform_(-bound, bound)

  
    # Defining the forward method is only necessary 
    # if you want to use a Trainer's .predict() method (optional)
    def forward(self, x, dx):
        return self.model(x, dx)
        
    # A common forward step to compute the loss and labels
    # this is used for training, validation, and testing below
    def _shared_step(self, batch):
        features, datasets, true_labels = batch["encoding"], batch["dataset_num"], batch["target"]

        if self.dca:
            features = torch.hstack((features, batch["dca"].unsqueeze(1)))
        
        # Convert class labels for CORAL ------------------------
        levels = levels_from_labelbatch(
            true_labels, num_classes=self.model.num_classes)
        # -------------------------------------------------------

        logits = self(features, datasets)

        # CORAL Loss --------------------------------------------
        # A regular classifier uses:
        # loss = torch.nn.functional.cross_entropy(logits, true_labels)
        loss = coral_loss(logits, levels.type_as(logits))
        # -------------------------------------------------------

        # CORAL Prediction to label -----------------------------
        # A regular classifier uses:
        # predicted_labels = torch.argmax(logits, dim=1)
        probas = torch.sigmoid(logits)
        predicted_labels = proba_to_label(probas)
        # -------------------------------------------------------
        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("train_loss", loss)
        self.train_mae(predicted_labels, true_labels)
        self.log("train_mae", self.train_mae, on_epoch=True, on_step=False)
        return loss  # this is passed to the optimzer for training

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("valid_loss", loss)
        self.valid_mae(predicted_labels, true_labels)
        self.log("valid_mae", self.valid_mae,
                 on_epoch=True, on_step=False, prog_bar=True)

    def test_step(self, batch, batch_idx):
        _, true_labels, predicted_labels = self._shared_step(batch)
        self.test_mae(predicted_labels, true_labels)
        self.log("test_mae", self.test_mae, on_epoch=True, on_step=False)

    def predict_step(self, batch, batch_idx, *args, **kwargs):
        #_, true_labels, predicted_labels = self._shared_step(batch)
        features, datasets, true_labels = \
                batch["encoding"], batch["dataset_num"], batch["target"]
        if self.dca:
            features = torch.hstack((features, batch["dca"].unsqueeze(1)))

        with torch.no_grad():
            logits = self(features, datasets)
            probas = torch.sigmoid(logits)
            predicted_labels = proba_to_label(probas)
        #probas = torch.empty(predicted_labels.shape[0], 3)
        return probas, true_labels, predicted_labels

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer



def create_model(model_config):
    """ Takes a model config object and returns a model that can run.
        
        Note: Unlike sklearn models, Pytorch models do not do automatic CV
    """
    mc = model_config
    model = None
    if model_config.model_name == "PytorchRegression":
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
    elif model_config.model_name == "MLPLightning":
        raise NotImplementedError
    else:
        raise ValueError(f"{model_name} must be one of {model_names}")
    return model

def get_metrics_df(csv_filename):
    metrics = pd.read_csv(csv_filename)
    aggreg_metrics = []
    agg_col = "epoch"
    for i, dfg in metrics.groupby(agg_col):
        agg = dict(dfg.mean())
        agg[agg_col] = i
        aggreg_metrics.append(agg)
    return pd.DataFrame(aggreg_metrics)

def plot_metrics_df(df_metrics, png_filename):
    df_metrics.plot.line(
            x="epoch", 
            y=[c for c in df_metrics.columns if (
                    c.endswith("_loss") or c.endswith("_mae"))],
            figsize=(10,6)).get_figure().savefig(png_filename)


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

    #model = create_model(model_config = mc)
    #model.xavier_init()
    #for p in model.named_parameters():
    #    print(p)

    SYNTHETIC_DATA = True

    additional_features = 0
    if mc.design_matrix["dca"]:
        additional_features += 1
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
    
    dm = None
    if SYNTHETIC_DATA:
        dm = Oct22SyntheticDataModule(size=1000, round_nums=(1,2,3))
    #dm.setup()
    #dl = dm.train_dataloader()
    #print(len(dl))
    #x = next(iter(dl))

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

    logger = CSVLogger(save_dir="logs/", name="mlp-lightning")
    trainer = pl.Trainer(max_epochs=args.num_epochs, log_every_n_steps=10, logger=logger)
    if dm is None:
        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    else:
        trainer.fit(model, datamodule=dm)


    df_metrics  = get_metrics_df(f"{trainer.logger.log_dir}/metrics.csv")
    plot_metrics_df(df_metrics, png_filename=args.output_dir / f"{mc.uuid}.losses.png")

    # get testing data
    test_dl = None
    if dm is None:
        test_ds = Pytorch_Oct22DataSet.create_from_args(
                    args=args, dataset_name=args.test_name)
        test_dl = DataLoader(test_ds, batch_size=args.batch_size, 
                                num_workers=args.num_workers)
        logging.info(f"len(test_ds) : {len(test_ds)}")
    else:
        test_dl = dm.predict_dataloader()
    probas, true_labels, predicted_labels = list(zip(*trainer.predict(model, test_dl)))
    probas = torch.cat(probas, dim=0)
    true_labels = torch.hstack(true_labels)
    predicted_labels = torch.hstack(predicted_labels)
    print(predicted_labels)
    print("Test set MAE=", 
            torchmetrics.MeanAbsoluteError()(predicted_labels, true_labels))
    print("All ones MAE=",
            torchmetrics.MeanAbsoluteError()(torch.ones(true_labels.shape), true_labels))
    np.savetxt(args.output_dir / f"{mc.uuid}.preds.txt", 
            predicted_labels)
    cum_probs = np.hstack([
                    np.ones((probas.shape[0], 1)), 
                    probas, 
                    np.zeros((probas.shape[0], 1))])
    probs = -np.diff(cum_probs)
    np.savetxt(args.output_dir / f"{mc.uuid}.probs.txt", probs)
    if args.save_model:
        trainer.save_checkpoint(args.output_dir / f"{mc.uuid}.ckpt")

