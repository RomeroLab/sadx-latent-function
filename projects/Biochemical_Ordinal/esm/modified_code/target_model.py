import argparse
import logging
import random
import shutil
import warnings
from argparse import ArgumentParser
from os.path import join
# from configargparse import ArgumentParser
from typing import Optional, Union

import finetuning_scheduler
import pytorch_lightning as pl
import wandb
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichModelSummary, TQDMProgressBar, \
    LearningRateMonitor, Checkpoint, StochasticWeightAveraging, ModelSummary
from finetuning_scheduler import FinetuningScheduler

import finetuning
import models
import shared_model
import utils
from datamodules import DMSDataModule
from finetuning import AnyFinetuning
from tasks import DMSTask
import analysis as an

logging.basicConfig(level=logging.INFO)


def init_basic_callbacks(enable_progress_bar: bool = True) -> list[Callback]:
    callbacks = [
        ModelSummary(max_depth=-1),
        LearningRateMonitor(),
    ]
    if enable_progress_bar:
        callbacks.append(TQDMProgressBar(refresh_rate=10))
    return callbacks


def init_callbacks(args, log_dir, dm) -> list[Callback]:

    # get the basic callbacks
    callbacks = init_basic_callbacks(args.enable_progress_bar)

    # determine quantity to monitor w/ checkpoint and early stopping callbacks
    # if we are using early stopping, then we also want the checkpoint callback to monitor the same quantity
    # because the checkpoint callback is used to reload the checkpoint w/ best quantity
    if args.es_monitor != "auto" and args.ckpt_monitor != "auto" and args.es_monitor != args.ckpt_monitor:
        warnings.warn("Monitors es_monitor and ckpt_monitor are set to different values, which means we may early stop "
                      "at a certain epoch but use a checkpoint from a different epoch based on each monitored "
                      "quantity. This is probably unintentional.")

    es_monitor = None
    if args.early_stopping:
        if args.es_monitor == "auto":
            if not dm.has_val_set:
                warnings.warn("Using train loss for early stopping because no validation set provided")
            es_monitor = "val_loss" if dm.has_val_set else "train_loss_epoch"
        else:
            es_monitor = "val_loss" if args.es_monitor == "val" else "train_loss_epoch"

    # ckpt monitor defaults to the same as early stopping monitor when set to auto
    if args.ckpt_monitor == "auto":
        ckpt_monitor = es_monitor
    elif args.ckpt_monitor == "val":
        ckpt_monitor = "val_loss"
    elif args.ckpt_monitor == "train":
        ckpt_monitor = "train_loss_epoch"
    else:
        raise ValueError("unsupported ckpt_monitor {}".format(args.ckpt_monitor))

    # monitor the best train or val loss depending on the monitored metric
    if ckpt_monitor is not None:
        callbacks.append(shared_model.BestMetricLogger(metric=ckpt_monitor, mode="min"))

    # set up model checkpoint and early stopping callbacks
    # if we are finetuning with the finetuning scheduler, then we need to set FTSModelCheckpoint instead
    # and that will be set below
    if not args.finetuning or (args.finetuning and args.finetuning_strategy != "fts"):
        checkpoint_callback = ModelCheckpoint(
            monitor=ckpt_monitor,
            mode="min",
            every_n_epochs=1,
            dirpath=join(log_dir, "checkpoints"),
            save_last=True
        )
        callbacks.append(checkpoint_callback)

    if args.early_stopping:
        early_stop_callback = EarlyStopping(
            monitor=es_monitor,
            min_delta=args.es_min_delta,
            patience=args.es_patience,
            verbose=True,
            mode='min'
        )
        callbacks.append(early_stop_callback)

    # set up finetuning callbacks
    if args.finetuning and args.finetuning_strategy == "extract":
        # for the 'extract' strategy, just need to freeze the backbone layers
        callbacks.append(finetuning.BackboneFreezer())

    # backbone finetuning callback
    elif args.finetuning and args.finetuning_strategy == "backbone":
        if args.early_stopping:
            warnings.warn("Using epoch-based backbone finetuning with early stopping enabled. It is possible early "
                          "stopping triggers before the backbone is unfrozen, thus no finetuning would take place. "
                          "Consider using in combination with min_epochs so early stopping only triggers during "
                          "the finetuning phase, if that's what you're going for.")
        finetuning_callback = AnyFinetuning(
            unfreeze_backbone_at_epoch=args.unfreeze_backbone_at_epoch,
            always_align_lr=args.backbone_always_align_lr,
            backbone_initial_ratio_lr=args.backbone_initial_ratio_lr,
            backbone_initial_lr=args.backbone_initial_lr,
            should_align=True,
            train_bn=args.train_bn,
            verbose=False,
            backbone_access_string="model.model.backbone")
        callbacks.append(finetuning_callback)

    elif args.finetuning and args.finetuning_strategy == "fts":
        # finetuning scheduler callback
        fts_callback = FinetuningScheduler(
            ft_schedule=args.fts_schedule_fn,
            epoch_transitions_only=True,
            restore_best=args.fts_restore_best,
            gen_ft_sched_only=False,
            apply_lambdas_new_pgs=True
        )
        callbacks.append(fts_callback)

        checkpoint_callback = finetuning_scheduler.FTSCheckpoint(
            monitor=ckpt_monitor,
            mode="min",
            every_n_epochs=1,
            dirpath=join(log_dir, "checkpoints"),
            save_last=True
        )
        callbacks.append(checkpoint_callback)

    # stochastic weight averaging callback
    if args.swa:
        swa_callback = StochasticWeightAveraging(swa_lrs=args.swa_lr, swa_epoch_start=args.swa_epoch_start)
        callbacks.append(swa_callback)

    return callbacks


def log_end_of_training_stats():
    # log end of training info about which model is used for evaluation
    # edit: got to figure out how to log outside of loop and what my goal is... maybe just a txt file like hparams?
    # task.log("best_model_path", ckpt_callback.best_model_path)
    # task.log("best_model_score", ckpt_callback.best_model_score)
    # parse the best model path and log the epoch
    # task.log("best_model_epoch", int(ckpt_callback.best_model_path.split("-")[0].split("=")[-1]))
    # if trainer.early_stopping_callback is not None:
    #     task.log("stopped_early", trainer.early_stopping_callback.stopped_epoch != 0)
    # else:
    #     task.log("stopped_early", False)
    pass


def error_checking(args):
    """ errors and warnings """
    if models.Model[args.model_name].transfer_model:
        if not args.finetuning:
            warnings.warn("Using a transfer learning model, but finetuning is disabled. This means the whole model "
                          "will be trained end-to-end. There are no frozen layers or finetuning.")


def log_config(loggers, args):
    """ log additional config to make processing runs easier
        note this has to be logged separately for each logger because it's being logged outside the LightningModule """

    config = {"eval_type": an.get_eval_type(args.split_dir),
              "train_size": an.get_train_size(args.split_dir),
              "split_rep_num": an.get_split_rep_num(args.split_dir),
              "seed": args.seed}

    wandb_logger = loggers[0]
    wandb_logger.experiment.config.update(config, allow_val_change=True)


def es_warning(ckpt_callback, es_callback):
    """ early stopping vs. model checkpoint epoch warning """
    ckpt_epoch = int(ckpt_callback.best_model_path.split("-")[0].split("=")[-1])

    # early stopping is optional so check if None
    if es_callback is not None:
        es_epoch = es_callback.stopped_epoch - es_callback.patience
        # this check won't be needed in future ver of Lightning https://github.com/Lightning-AI/lightning/issues/14353
        if es_epoch >= 0 and (ckpt_epoch != es_epoch):
            # note: also checking to make sure es_epoch != 1... because if it does == 1, then ES didn't trigger,
            # so no need to compare ES epoch to ckpt epoch, ES epoch is just a placeholder in this case
            warnings.warn(
                f"EarlyStopping callback recorded the best epoch as epoch {es_epoch}, however, ModelCheckpoint "
                f"callback recorded epoch {ckpt_epoch}. This is likely due to the min_delta, which EarlyStopping "
                f"supports but ModelCheckpoint does not (Lightning should support it in future version). The "
                f"model from epoch {es_epoch} has a val loss of {es_callback.best_score}, and the model from "
                f"epoch {ckpt_epoch} has a val loss of {ckpt_callback.best_model_score}. Test metrics will "
                f"be calculated with the ModelCheckpoint saved checkpoint from epoch {ckpt_epoch}."
            )


def verify_set_seed(args):
    if args.seed is None:
        # you are entitled to a random seed
        # if you do not specify a seed, one will be specified for you
        args.seed = random.randint(100000000, 999999999)
        print("Random seed not specified, using: {}".format(args.seed))


def metrics_df_to_dict(df, suffix=""):
    result_dict = {}
    for idx in df.index:
        for col in df.columns:
            key = f"metrics_{idx}_{col}{suffix}"
            result_dict[key] = df.loc[idx, col]
    return result_dict


def log_metrics(raw_preds, dm, log_dir, trainer, args):
    predictions_d = shared_model.save_predictions(raw_preds, dm, log_dir, save_format="npy")
    shared_model.save_scatterplots(dm, predictions_d, log_dir)
    metrics_df = shared_model.save_metrics_custom(dm, predictions_d, log_dir)
    shared_model.plot_losses(log_dir)

    # log metrics_custom to wandb
    wandb.log(metrics_df_to_dict(metrics_df))

    if args.save_last_metrics:
        # saves a metric_custom_last.txt file with the metrics computed on the last checkpoint
        # run the test metrics via PTL for wandb :)
        raw_preds = trainer.predict(ckpt_path="last", datamodule=dm, return_predictions=True)
        predictions_d = shared_model.save_predictions(raw_preds, dm, log_dir, save_format="npy", suffix="_last")
        shared_model.save_scatterplots(dm, predictions_d, log_dir, suffix="_last")
        metrics_df = shared_model.save_metrics_custom(dm, predictions_d, log_dir, suffix="_last")
        wandb.log(metrics_df_to_dict(metrics_df, suffix="_last"))


def main(args: argparse.Namespace):

    error_checking(args)
    verify_set_seed(args)

    pl.seed_everything(args.seed)

    # get the uuid and log directory for this run
    my_uuid, log_dir = shared_model.create_log_dir(args.log_dir_base, args.uuid)

    # save arguments to the log directory
    utils.save_args(vars(args), join(log_dir, "args.txt"), ignore=["cluster", "process"])

    # set up logger callbacks for training
    loggers = shared_model.init_loggers(log_dir, my_uuid, args.wandb_online, args.wandb_project)

    # log some config parameters for wandb to make exploring runs easier
    log_config(loggers, args)

    # load data and split via the datamodule
    dm = DMSDataModule(**vars(args))

    # create the model and task
    task = DMSTask(num_tasks=dm.num_tasks,
                   num_tokens=dm.num_tokens,
                   aa_seq_len=dm.aa_seq_len,
                   aa_encoding_len=dm.aa_encoding_len,
                   seq_encoding_len=dm.seq_encoding_len,
                   pdb_fns=dm.unique_pdb_fns,
                   example_input_array=dm.example_input_array,
                   **vars(args))

    callbacks = init_callbacks(args, log_dir, dm)

    # set up wandb to log gradients, parameter histograms
    if args.wandb_log_grads:
        loggers[0].watch(task, log="all", log_freq=args.grad_log_freq)

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


def add_target_args(parent_parser):
    """ args specific to target model training and finetuning (shared with ESM...) """
    p = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

    # random seed
    p.add_argument("--seed", help="random seed to use with pytorch lightning seed_everything"
                                  "not specifying a seed will use a random seed and record it to args "
                                  "for future runs",
                   default=None, type=int)

    # early stopping
    p.add_argument("--early_stopping", help="set to enable early stopping", action="store_true")
    p.add_argument("--es_monitor", help="which loss to monitor", default="auto", choices=["train", "val", "auto"])
    p.add_argument("--es_patience", help="number of epochs allowance for early stopping", type=int, default=5)
    p.add_argument("--es_min_delta", help="min by which the loss must decrease to be considered an improvement",
                   type=float, default=0.001)

    # checkpoint callback monitoring metric
    # mostly meant for when early stopping = False, but still want to choose best model based on metric
    p.add_argument("--ckpt_monitor", help="which loss to monitor for ckpt",
                   default="auto", choices=["train", "val", "auto"])

    # fine tuning
    p.add_argument("--finetuning", action="store_true", default=False)
    p.add_argument("--finetuning_strategy", type=str, default="backbone", choices=["backbone", "extract", "fts"])

    # for 'backbone' finetuning strategy
    p.add_argument("--unfreeze_backbone_at_epoch", type=int, default=10)
    p.add_argument("--train_bn", help="whether to train batchnorm in backbone", action="store_true")
    p.add_argument("--backbone_always_align_lr", action="store_true", default=False)
    p.add_argument("--backbone_initial_ratio_lr", type=float, default=0.1)
    p.add_argument("--backbone_initial_lr", type=float, default=None)

    # for 'FTS' finetuning strategy
    p.add_argument("--fts_schedule_fn", type=str, default=None, help="path to the schedule file for FTS finetuning")
    p.add_argument("--fts_restore_best", action="store_true", default=False)

    # stochastic weight averaging
    p.add_argument('--swa', help="set to enable stochastic weight averaging", action="store_true")
    p.add_argument('--swa_epoch_start', type=int, default=None)
    p.add_argument('--swa_lr', type=float, default=0.0001)

    # save the metrics for the last checkpoint
    p.add_argument("--save_last_metrics", help="set to save metrics for the last checkpoint", action="store_true")

    return p


if __name__ == "__main__":
    parser = ArgumentParser(add_help=True)

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
    parser.add_argument("--wandb_log_grads",
                        default=False,
                        action="store_true",
                        help="whether to log gradients and parameter histograms to weights&biases")
    parser.add_argument("--grad_log_freq",
                        default=500,
                        type=int,
                        help="log frequency for gradients")
    parser.add_argument("--delete_checkpoints",
                        action="store_true",
                        default=False)

    # add data specific args
    parser = DMSDataModule.add_data_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = pl.Trainer.add_argparse_args(parser)

    # figure out which model to use
    # need to have this additional argument parser line to add the fromfile_prefix_chars
    # this lets us specify the model_name in the file along with model specific args
    parser = ArgumentParser(parents=[parser], fromfile_prefix_chars='@', add_help=False)

    # special model choice "transfer_model" signifies we are loading a backbone from a checkpoint
    # transfer_model_keyword = "transfer_model"
    model_choices = [m.name for m in list(models.Model)]  # + [transfer_model_keyword]
    parser.add_argument("--model_name", type=str, choices=model_choices)

    # grab the model_name
    temp_args, _ = parser.parse_known_args()
    # temp_args, _ = parser.parse_args()

    # add task-specific args
    parser = DMSTask.add_model_specific_args(parser)

    # add model-specific args
    add_args_op = getattr(models.Model[temp_args.model_name].cls, "add_model_specific_args", None)
    if callable(add_args_op):
        parser = add_args_op(parser)

    # finally, make sure we can use args from file (can't do this before because it gets overwritten)
    parser = ArgumentParser(parents=[parser], fromfile_prefix_chars='@', add_help=False)

    parsed_args = parser.parse_args()

    main(parsed_args)
