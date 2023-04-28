""" Run the ESM model fine-tuning """
import argparse
import collections
import shutil
# workaround for downloading models from torchhub on condor
import ssl
from argparse import ArgumentParser
from os.path import join
from typing import Optional, Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import models
import shared_model
from target_model import verify_set_seed, log_config, init_callbacks, es_warning, log_metrics, add_target_args
import utils
from esm_shared import ESMSequenceRep
from esm_shared_ordinal import ESMDataModule

ssl._create_default_https_context = ssl._create_unverified_context


class ESMTransferModel(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        p = ArgumentParser(parents=[parent_parser], add_help=False)

        # top net args
        p.add_argument("--dropout_after_backbone", action="store_true")
        p.add_argument("--dropout_after_backbone_rate", type=float, default=0.1)
        p.add_argument("--top_net_type", type=str, default="linear", choices=["linear", "nonlinear"])
        p.add_argument("--top_net_hidden_nodes", type=int, default=256)
        p.add_argument("--top_net_use_batchnorm", action="store_true")
        p.add_argument("--top_net_use_dropout", action="store_true")
        p.add_argument("--top_net_dropout_rate", type=float, default=0.1)

        return p

    def __init__(self,
                 esm_sequence_rep_model: nn.Module,
                 esm_embedding_dim: int,
                 dropout_after_backbone: bool = False,
                 dropout_after_backbone_rate: float = 0.1,
                 top_net_type: str = "linear",
                 top_net_hidden_nodes: int = 256,
                 top_net_use_batchnorm: bool = False,
                 top_net_use_dropout: bool = False,
                 top_net_dropout_rate: float = 0.1,
                 *args, **kwargs):

        super().__init__()

        layers = collections.OrderedDict()
        layers["backbone"] = esm_sequence_rep_model

        # add dropout after backbone if specified
        if dropout_after_backbone:
            layers["dropout"] = nn.Dropout(dropout_after_backbone_rate)

        layers["flatten"] = nn.Flatten(start_dim=1)

        # create a new prediction layer on top of the backbone
        if top_net_type == "linear":
            layers["prediction"] = nn.Linear(in_features=esm_embedding_dim, out_features=1)
        elif top_net_type == "nonlinear":
            fc_block = models.FCBlock(in_features=esm_embedding_dim,
                                      num_hidden_nodes=top_net_hidden_nodes,
                                      use_batchnorm=top_net_use_batchnorm,
                                      use_dropout=top_net_use_dropout,
                                      dropout_rate=top_net_dropout_rate)

            pred_layer = nn.Linear(in_features=top_net_hidden_nodes, out_features=1)
            layers["prediction"] = nn.Sequential(fc_block, pred_layer)
        else:
            raise ValueError("Unexpected type of top net layer: {}".format(top_net_type))

        self.model = nn.Sequential(layers)

    def forward(self, x):
        return self.model(x)


class ESMTrainingTask(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        return shared_model.OptimizerConfig.add_model_specific_args(parent_parser)

    def __init__(self,
                 model: nn.Module,
                 # optimizer params
                 optimizer: str = "adamw",
                 weight_decay: float = 0.01,
                 learning_rate: float = 0.0001,
                 lr_scheduler: str = "constant",
                 warmup_steps: float = .02,
                 phase2_lr_ratio: float = 1.0,
                 # other
                 example_input_array: Optional[Any] = None,
                 save_hyperparams=True,
                 *args, **kwargs):

        super().__init__()

        if save_hyperparams:
            # don't re-save hyperparameters when initializing a phase2 task
            # note: don't use self.hparams if save_hyperparams is False! it won't be set
            self.save_hyperparameters(ignore=["model", "example_input_array"])

        self.model = model
        self.example_input_array = example_input_array

        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.warmup_steps = warmup_steps
        self.phase2_lr_ratio = phase2_lr_ratio

        self.test_pearson = torchmetrics.PearsonCorrCoef()
        self.test_spearman = torchmetrics.SpearmanCorrCoef()

    @staticmethod
    def custom_mse_loss(inputs, targets):
        return torch.sum(torch.mean((inputs - targets) ** 2, dim=0))

    def _shared_step(self, batch, batch_idx, compute_loss=True):
        outputs = self(batch)
        if compute_loss:
            loss = self.custom_mse_loss(outputs, batch["targets"])
            return outputs, loss
        else:
            return outputs, None

    def training_step(self, data_batch, batch_idx):
        outputs, loss = self._shared_step(data_batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, data_batch, batch_idx):
        outputs, loss = self._shared_step(data_batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, data_batch, batch_idx):
        # compute and log the loss
        outputs, loss = self._shared_step(data_batch, batch_idx)
        self.log("test_loss", loss)

        # other metrics
        labels = data_batch["targets"]
        self.test_pearson(torch.squeeze(outputs, dim=-1), torch.squeeze(labels, dim=-1))
        self.test_spearman(torch.squeeze(outputs, dim=-1), torch.squeeze(labels, dim=-1))
        self.log("test_pearson", self.test_pearson, on_step=False, on_epoch=True)
        self.log("test_spearman", self.test_spearman, on_step=False, on_epoch=True)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        outputs, _ = self._shared_step(batch, batch_idx, compute_loss=False)
        return outputs

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))

        optimizer_config = shared_model.OptimizerConfig(self.optimizer,
                                                        self.weight_decay,
                                                        self.learning_rate,
                                                        self.lr_scheduler,
                                                        self.warmup_steps,
                                                        self.phase2_lr_ratio)

        return optimizer_config.get_optimizer_config(trainable_parameters, self.trainer.estimated_stepping_batches)


def init_esm_task_and_dm(args):
    # parse the representation layer from the esm_version name
    rep_layer = int(args.esm_version.split("_")[1][1:])
    print("Using ESM: {}".format(args.esm_version))
    print("Using rep layer: {}".format(rep_layer))

    esm_dim_map = {"esm2_t48_15B_UR50D": 5120,
                   "esm2_t36_3B_UR50D": 2560,
                   "esm2_t33_650M_UR50D": 1280,
                   "esm2_t30_150M_UR50D": 640,
                   "esm2_t12_35M_UR50D": 480,
                   "esm2_t6_8M_UR50D": 320}

    # load ESM from PyTorch Hub
    esm_base, alphabet = torch.hub.load("facebookresearch/esm:main", args.esm_version, force_reload=False)

    # load the datamodule
    dm = ESMDataModule(alphabet=alphabet, predict_mode="all_sets", **vars(args))

    # load ESM module wrapper
    esm_seq_rep_model = ESMSequenceRep(esm_base, repr_layer=rep_layer, return_contacts=False, seq_len=dm.aa_seq_len)
    model = ESMTransferModel(esm_seq_rep_model, esm_embedding_dim=esm_dim_map[args.esm_version], **vars(args))

    # create a pytorch lightning wrapper for easy model train, eval, gpu transfer, predict
    task = ESMTrainingTask(model, example_input_array=dm.example_input_array, **vars(args))

    return task, dm


def main(args: argparse.Namespace):

    # set the torch hub cache directory
    torch.hub.set_dir(args.hub_dir)

    # make sure random seed is set
    verify_set_seed(args)
    pl.seed_everything(args.seed)

    # get the uuid and log directory for this run
    my_uuid, log_dir = shared_model.create_log_dir(args.log_dir_base, args.uuid)

    # save arguments to the log directory
    utils.save_args(vars(args), join(log_dir, "args.txt"), ignore=["cluster", "process"])

    # set up logger callbacks for training
    loggers = shared_model.init_loggers(log_dir, my_uuid, args.wandb_online, args.wandb_project)
    ## ** Commented out by Sameer ** ##
    ## log some config parameters for wandb to make exploring runs easier
    #log_config(loggers, args)

    # load the ESM task and datamodule
    task, dm = init_esm_task_and_dm(args)

    callbacks = init_callbacks(args, log_dir, dm)

    # htcondor was giving me slots with more than 1 gpu, which was causing problems
    # so if we are running on condor, specify devices = 1
    # if running locally, use devices = auto which should select available CPU cores
    devices = "auto" if args.cluster == "local" else 1
    trainer: pl.Trainer = pl.Trainer.from_argparse_args(args,
                                                        default_root_dir=log_dir,
                                                        callbacks=callbacks,
                                                        logger=loggers,
                                                        accelerator="auto",
                                                        devices=devices)

    trainer.fit(task, datamodule=dm)

    # assuming there is a checkpoint callback (all target models should have this)
    ckpt_callback: Optional[ModelCheckpoint] = trainer.checkpoint_callback
    es_callback: Optional[EarlyStopping] = trainer.early_stopping_callback

    # warn if the ckpt epoch doesn't match the es epoch (won't be needed in future ver of lightning)
    es_warning(ckpt_callback, es_callback)

    # run test set and save metrics and losses
    test_metrics = trainer.test(ckpt_path="best", datamodule=dm)

    # save metrics computed by pytorch lightning along w/ the specific checkpoint used to compute those metrics
    shared_model.save_metrics_ptl(ckpt_callback.best_model_path, ckpt_callback.best_model_score, test_metrics, log_dir)

    # save predictions, scatterplots, and custom metrics. plot train loss vs. val loss
    raw_preds = trainer.predict(ckpt_path="best", datamodule=dm, return_predictions=True)

    # log end of training metrics
    log_metrics(raw_preds, dm, log_dir, trainer, args)

    # delete the checkpoints folder if asked
    if args.delete_checkpoints:
        shutil.rmtree(join(log_dir, "checkpoints"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument("--esm_version",
                        help="torch hub string for the esm model to use",
                        type=str,
                        default="esm2_t12_35M_UR50D")
    parser.add_argument("--hub_dir",
                        help="cache directory for pre-trained models from torch hub",
                        type=str,
                        default="output/esm_pretrained_models")

    # Program args
    parser = add_target_args(parser)

    # HTCondor args
    parser.add_argument("--cluster",
                        help="cluster (when running on HTCondor)",
                        type=str,
                        default="local")
    parser.add_argument("--process",
                        help="process (when running on HTCondor)",
                        type=str,
                        default="local")
    parser.add_argument("--github_tag",
                        help="github tag for current run",
                        type=str,
                        default="no_github_tag")

    # additional args
    parser.add_argument("--log_dir_base",
                        help="log directory base",
                        type=str,
                        default="output/training_logs")
    parser.add_argument("--uuid",
                        help="model uuid to resume from or custom uuid to use from scratch",
                        type=str,
                        default=None)
    parser.add_argument("--wandb_online",
                        action="store_true",
                        default=False)
    parser.add_argument("--wandb_project",
                        type=str,
                        default="rtl_target")
    parser.add_argument("--experiment",
                        type=str,
                        default="default",
                        help="dummy arg to make wandb tracking and filtering easier")
    parser.add_argument("--delete_checkpoints",
                        action="store_true",
                        default=False)

    # add all the available trainer options to argparse
    parser = pl.Trainer.add_argparse_args(parser)

    # add data specific args
    parser = ESMDataModule.add_data_specific_args(parser)

    # add model specific args
    parser = ESMTransferModel.add_model_specific_args(parser)

    # add task specific args
    parser = ESMTrainingTask.add_model_specific_args(parser)

    parser = argparse.ArgumentParser(parents=[parser], fromfile_prefix_chars='@', add_help=False)
    main(parser.parse_args())

