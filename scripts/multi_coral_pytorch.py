import time
import random

import numpy as np
import pandas as pd

import torch
torch.use_deterministic_algorithms(True)
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import one_hot

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import torchmetrics

from sklearn.model_selection import train_test_split

from scipy.stats import spearmanr

from coral_multiple_layer import CoralMultipleLayer
from coral_pytorch.losses import coral_loss
from coral_pytorch.dataset import levels_from_labelbatch, proba_to_label


import utils
utils.add_projects_to_path()

from protein_utils.energy.potts import create_single_mutant

q = 20 # alphabet size

# Some of these can be changed by the command line arguments to this script
BATCH_SIZE = 64
NUM_EPOCHS = 50
#NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-6
NUM_WORKERS = 4



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

    def predict(self, x, dx=None):
        with torch.no_grad():
            # FIXME: this only predicts for a single x and not a batch
            return self.output_layer(self.model(x), torch.LongTensor([0])) 


# LightningModule that receives a PyTorch model as input
class LightningMLP(pl.LightningModule):
    def __init__(self, model, learning_rate, weight_decay):
        super().__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        # The inherited PyTorch module
        self.model = model

        # Save settings and hyperparameters to the log directory
        # but skip the model parameters
        self.save_hyperparameters(ignore=['model'])

        # Set up attributes for computing the MAE
        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        self.test_mae = torchmetrics.MeanAbsoluteError()
        
    # Defining the forward method is only necessary 
    # if you want to use a Trainer's .predict() method (optional)
    def forward(self, x, dx):
        return self.model(x, dx)
        
    # A common forward step to compute the loss and labels
    # this is used for training, validation, and testing below
    def _shared_step(self, batch):
        features, datasets, true_labels = batch
        
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

class DataModule(pl.LightningDataModule):

    def __init__(self, syn_data, syn_labels, syn_datasets, 
                    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                    L = None):
        """
            L is the length of the protein. The First L features will be
            one-hot encoded.  
        """
        super().__init__()
        self.syn_data = syn_data
        self.syn_labels = syn_labels
        self.label_map = {'H':3, "P":2, "L":1, "N":0}

        num_features = self.syn_data.shape[1]
        if not L: # L is not specified
            L = num_features
        if L > num_features:
            raise ValueError("L cannot be greater than the number of features")
        
        self.data_features =  \
                one_hot(torch.Tensor(self.syn_data[:, :L]).to(torch.long), 
                    num_classes=q).flatten(start_dim=1, end_dim=-1)
        if L < num_features:
            self.data_features = torch.hstack(
                    [self.data_features, 
                     torch.Tensor(self.syn_data[:, L:]).to(torch.float)])
        self.syn_labels = pd.Series(syn_labels)
        self.data_labels = torch.LongTensor(self.syn_labels.map(
                                self.label_map.__getitem__))
        self.datasets = torch.LongTensor(syn_datasets)

        self.batch_size = batch_size	
        self.num_workers = num_workers
        
    def prepare_data(self):
        return

    def setup(self, stage=None):
        # Split into approx
        # 70% train, 10% validation, 20% testing
        
        X_temp, X_test, X_dataset_temp, X_dataset_test, y_temp, y_test = \
                train_test_split(
                    self.data_features,
                    self.datasets,
                    self.data_labels,
                    test_size=0.2,
                    random_state=1,
                    stratify=self.data_labels)

        X_train, X_valid, X_dataset_train, X_dataset_valid, y_train, y_valid = \
                train_test_split(
                    X_temp,
                    X_dataset_temp,
                    y_temp,
                    test_size=0.1,
                    random_state=1,
                    stratify=y_temp)
                
        self.train = MyDataset(X_train, X_dataset_train, y_train)
        self.valid = MyDataset(X_valid, X_dataset_valid, y_valid)
        self.test = MyDataset(X_test, X_dataset_test, y_test)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size,
                            num_workers=self.num_workers)



class MyDataset(Dataset):

    def __init__(self, feature_array, dataset_array, label_array, dtype=np.float32):
        self.features = feature_array.float()
        self.datasets = dataset_array
        self.labels = label_array

    def __getitem__(self, index):
        inputs = self.features[index]
        dataset = self.datasets[index]
        label = self.labels[index]
        return inputs, dataset, label

    def __len__(self):
        return self.features.shape[0]

def load_synthetic_data(size=1000, round_nums=(1,2,3)):
    msas = []
    labels = []
    dataset_labels = []
    round_num = 1
    for round_num in round_nums:
        with np.load(f"../output/synthetic/msa_round_{round_num}_N_{size}.npz", 
                            allow_pickle=True) as data:
            msas.append(data["msa"])
            dataset_labels.append(np.repeat(round_num, data["msa"].shape[0]))
            labels.append(data["labels"])

    msas = np.concatenate(msas)
    labels = np.concatenate(labels)
    dataset_labels = np.concatenate(dataset_labels) - 1
    return msas, labels, dataset_labels


if __name__ == "__main__":
    seed=100
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    msas, labels, dataset_labels = load_synthetic_data() #all rounds
    # correlation 0.4503
    #msas, labels, dataset_labels = load_synthetic_data(round_nums=[1])
    # correlation 0.2096
    #msas, labels, dataset_labels = load_synthetic_data(round_nums=[2])
    # correlation 0.2056
    #msas, labels, dataset_labels = load_synthetic_data(round_nums=[3])
    # correlation 0.1642
    #msas, labels, dataset_labels = load_synthetic_data(round_nums=[1,2])
    # correlation 0.3695
    #msas, labels, dataset_labels = load_synthetic_data(round_nums=[1,3])
    # correlation 0.3578
    #msas, labels, dataset_labels = load_synthetic_data(round_nums=[2,3])
    # correlation 0.3354
    #msas, labels, dataset_labels = load_synthetic_data(size=2000, round_nums=[1])
    # correlation 0.3059
    #msas, labels, dataset_labels = load_synthetic_data(size=3000, round_nums=[1])
    # correlation 0.4243
    #msas, labels, dataset_labels = load_synthetic_data(size=10000, round_nums=[1])
    # correlation 0.6530
    #msas, labels, dataset_labels = load_synthetic_data(size=100000, round_nums=[1])
    # correlation 0.8596
    #print(msas.shape, labels.shape, dataset_labels.shape)
    data_module = DataModule(msas, labels, dataset_labels)

    _, L = msas.shape

    pytorch_model = BaseCoralModule(
        input_size=L*q,
        hidden_units=(100, 20),
        num_classes=4, num_datasets=dataset_labels.max()+1)

#    print(pytorch_model.output_layer.coral_weights)
#    print(pytorch_model.output_layer.coral_weights.weight.shape)
#    print(pytorch_model.output_layer.coral_bias.shape)
#
    lightning_model = LightningMLP(
        model=pytorch_model,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY)


    callbacks = [ModelCheckpoint(
        save_top_k=1, mode="min", monitor="valid_mae")]  # save top 1 model 
    logger = CSVLogger(save_dir="logs/", name="mlp-synthetic-msa")

    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        callbacks=callbacks,
        #progress_bar_refresh_rate=50,  # recommended for notebooks
        accelerator="auto",  # Uses GPUs or TPUs if available
        devices="auto",  # Uses all available GPUs/TPUs if applicable
        logger=logger,
        deterministic=True,
        log_every_n_steps=10)

    start_time = time.time()
    trainer.fit(model=lightning_model, datamodule=data_module)

    runtime = (time.time() - start_time)/60
    print(f"Training took {runtime:.2f} min in total.")

    pytorch_model.eval() # put model in evaluation mode
    mut_df = pd.read_csv("../output/synthetic_mutation_map.csv")
    wt_np = np.load("../output/synthetic/synthetic_wt.npy")
    def get_bin_N_score(i, a):
        x = one_hot(torch.LongTensor(create_single_mutant(i, a, wt_np))).flatten()
        with torch.no_grad():
            logits = pytorch_model.predict(x.float())
            probas = torch.sigmoid(logits) # probability of each class L, P, H
            # find out what is the probability of N and return log of that
            #print(probas)
            return torch.log(1 - probas.squeeze()[0]).item()

    bin_N_score = mut_df.apply(lambda x: get_bin_N_score(x['idx'], x['mut_aa']), axis=1) 

    print(spearmanr(bin_N_score, mut_df.energy))

    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")

    aggreg_metrics = []
    agg_col = "epoch"
    for i, dfg in metrics.groupby(agg_col):
        agg = dict(dfg.mean())
        agg[agg_col] = i
        aggreg_metrics.append(agg)

    df_metrics = pd.DataFrame(aggreg_metrics)
    x = df_metrics[["train_loss", "valid_loss"]].plot(
        grid=True, legend=True, xlabel='Epoch', ylabel='Loss')
    x.get_figure().savefig("../output/synthetic/synthetic_losses.png")
    x = df_metrics[["train_mae", "valid_mae"]].plot(
        grid=True, legend=True, xlabel='Epoch', ylabel='MAE')
    x.get_figure().savefig("../output/synthetic/synthetic_mae.png")
