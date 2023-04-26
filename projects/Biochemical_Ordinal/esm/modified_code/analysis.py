""" Some functions to make analysis of trained models easier """

import os
import shutil
import warnings
from os.path import join, isfile, isdir, basename
from typing import Union, Callable, Optional, Literal

import numpy as np
import pandas as pd
import scipy
import yaml
from tqdm import tqdm

import split_dataset as sd
import utils
import constants
import collections


def parse_log_dir_name(log_dir):
    """ simple function for parsing log dirs, if you need to parse the log dir name anywhere you should use this.
        because if you ever update the log dir format in regression.py, you just need to change this once instead of
        going through the whole codebase and searching for where you parse log dir names manually """

    # assuming no surprise underscores from net_file, etc
    tokens = basename(log_dir).split("_")
    parsed = {"date": tokens[3],
              "time": tokens[4],
              "cluster": tokens[1],
              "process": tokens[2],
              "wandb_project": tokens[5],
              "learning_rate": tokens[6],
              "batch_size": tokens[7],
              "uuid": tokens[8]}
    return parsed


def load_args(arg_file):
    """ load arguments from an args.txt file without invoking argparse (which would be better, but reasons)
        this doesn't do data types like loading from argparse would, but the argparse parser is more complicated
        to set up due to using pytorch lightning """
    args = dict()

    lines = utils.load_lines(arg_file)
    for i, line in enumerate(lines):
        if line.startswith("--"):
            # argument name (dictionary key)
            arg_name = line[2:]
            # argument value (dictionary value)
            if i+1 >= len(lines):
                # if the last arg is a flag, there won't be a next line to check
                arg_val = True
            elif lines[i+1].startswith("--"):
                # if the next line starts with "--", then the arg is just a flag and the value is true
                arg_val = True
            else:
                # otherwise the next line is the arg value
                # todo: if we have a list of items as an arg, we would need to check the next X lines until the next --
                arg_val = lines[i+1]
            args[arg_name] = arg_val

    return args


def flatten_dict(d, sep="/"):
    """ flattens a dictionary into a single level dictionary """
    out = dict()
    for k, v in d.items():
        if isinstance(v, dict):
            # if the value is a dictionary, recursively flatten it
            v = flatten_dict(v)
            for k2, v2 in v.items():
                # add the flattened dictionary to the output dictionary
                out[k + sep + k2] = v2
        else:
            # if the value is not a dictionary, just add it to the output dictionary
            out[k] = v
    return out


def load_yaml_args(arg_file):
    """ load arguments from pytorch lightning yaml file into a flat dictionary """

    # load the yaml file
    with open(arg_file, "r") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    # flatten the dictionary
    args = flatten_dict(args)

    return args


def load_predictions(log_dirs: Union[str, list[str]],
                     col_names: Optional[Union[str, list[str]]] = None,
                     ds: Optional[pd.DataFrame] = None,
                     return_set_names: bool = False):

    """ loads the predictions for a trained model from the training log directory.
        returns a dataframe with all the variants, scores, set names, and predictions.
        if passing in a list of log directories, this function expects they are all for the same dataset
        and are using the same train/tune/test set split. this function uses the first log directory in the list
         as the template from which it loads the dataset and train/tune/test split."""

    # TODO: pretty sure this won't work with SOURCE models because of dataset loading (database instead of csv, etc)

    # if we're passed in a single log directory as a string, convert it to a list for consistent handling
    if isinstance(log_dirs, str):
        log_dirs = [log_dirs]

    # if no column names were specified, automatically generate based on log_dirs indexing
    if col_names is None:
        # using a simple naming scheme of "prediction_index", but maybe I should use the UUID of the model instead?
        col_names = ["prediction"] if len(log_dirs) == 1 else ["prediction_{}".format(i) for i in range(len(log_dirs))]
    # passed in a single column name as a string, convert it to a list for consistent handling
    elif isinstance(col_names, str):
        col_names = [col_names]

    # ensure we have a col name for each log dir
    if len(log_dirs) != len(col_names):
        raise ValueError("the number of log dirs {} should equal the number of column names {}".format(
            len(log_dirs), len(col_names)))

    all_set_names = set()
    for i, log_dir in enumerate(log_dirs):
        if not isdir(log_dir):
            raise FileNotFoundError("couldn't find log directory {}".format(log_dir))

        args_fn = join(log_dir, "args.txt")
        if not isfile(args_fn):
            raise FileNotFoundError("couldn't find args.txt inside of log directory {}".format(log_dir))

        # load the arguments used to train the model
        args = load_args(args_fn)

        # todo: RosettaTL doesn't currently save the split in the log directory, but it may in the future
        # if the split dir isn't in the same log directory, grab it from the args file
        if isdir(join(log_dir, "split")):
            split_dir = join(log_dir, "split")
        elif isdir(args["split_dir"]):
            split_dir = args["split_dir"]
        else:
            raise FileNotFoundError("couldn't find log directory {} or {}".format(log_dir, args["split_dir"]))

        # grab the predictions fns from the predictions dir
        predictions_dir = join(log_dir, "predictions")
        predictions_fns = [join(predictions_dir, f) for f in os.listdir(predictions_dir) if
                           f.endswith("_predictions.npy") or f.endswith("_predictions.txt")]
        if len(predictions_fns) == 0:
            raise FileNotFoundError("couldn't find any predicted scores files")

        # load the full dataset used to train the model
        if ds is None:
            ds = utils.load_dataset(ds_name=args["ds_name"])
        else:
            ds = ds.copy()

        # load the split used to train the model and add a column to the dataframe with the set_name
        split = sd.load_split_dir(split_dir, filetype="npy")

        # maintain a list of all the set names we've seen (used downstream when calculating additional metrics)
        for set_name in split.keys():
            all_set_names.add(set_name)

        if "set_name" not in ds.columns:
            ds["set_name"] = np.nan
            for set_name, idxs in split.items():
                ds.iloc[idxs, ds.columns.get_loc("set_name")] = set_name
        else:
            # check to see if the train/tune/test split is the same
            for set_name, idxs in split.items():
                if not np.all(ds.iloc[idxs, ds.columns.get_loc("set_name")] == set_name):
                    raise ValueError("detected different train/tune/test splits among given log dirs")

        # now go through each set of predictions and add to the appropriate place in the dataframe
        ds[col_names[i]] = np.nan
        for pfn in predictions_fns:
            set_name = basename(pfn).split("_")[0]

            # todo: fix inconsistency w/ set_name "test" and "stest"
            # need this check because METL saves the predictions as "test.txt" even though the set name is "stest"
            # this occurs because the set name is "auto" in the args file, which is resolved to "stest" in the code
            # but then when the predictions are saved, the set name used is "test" instead of "stest"
            if set_name == "test" and "test" not in split and "stest" in split:
                set_name = "stest"

            if pfn.endswith(".txt"):
                ds.iloc[split[set_name], ds.columns.get_loc(col_names[i])] = np.loadtxt(pfn)
            elif pfn.endswith(".npy"):
                ds.iloc[split[set_name], ds.columns.get_loc(col_names[i])] = np.load(pfn)
            else:
                raise ValueError("unknown predictions file extension")

    if return_set_names:
        return ds, all_set_names
    else:
        return ds


def load_args_df(log_dir, args_fn="args.txt", args_type="custom"):
    """ loads the args used to train a model in the form of a dataframe.
        this is not meant to be used to re-train models, rather it's for analysis purposes.
        the args may contain extra info that aren't passed in to regression.py, like the UUID """

    args_fn = join(log_dir, args_fn)
    if not isfile(args_fn):
        # raise FileNotFoundError("couldn't find args.txt inside of log directory {}".format(args_fn))
        return None

    # load the arguments used to train this model
    if args_type == "custom":
        args = load_args(args_fn)
    elif args_type == "lightning":
        args = load_yaml_args(args_fn)
    else:
        raise ValueError("unknown args_type {}".format(args_type))

    # add the UUID and log_dir to the args
    args["log_dir"] = log_dir

    # the uuid is already in lightning args, only need to add it to custom args
    if args_type == "custom":
        log_dir_parsed = utils.parse_log_dir_name(log_dir)
        args["uuid"] = log_dir_parsed["uuid"]

        # also add the cluster and process, which aren't in the arguments
        # args["cluster"] = log_dir_parsed["cluster"]
        # args["process"] = log_dir_parsed["process"]

    args_df = pd.DataFrame.from_dict(args, orient="index").T

    # move uuid and other index-like items to first column
    # c2m = ["uuid", "cluster", "process", "log_dir", "wandb_project"]
    c2m = ["uuid", "log_dir", "wandb_project"]
    args_df = args_df.reindex(columns=c2m + [c for c in args_df.columns if c not in c2m])

    return args_df


def load_metrics(log_dir, eval_fn="metrics_custom.txt", eval_type="custom"):
    eval_fn = join(log_dir, eval_fn)

    if not isfile(eval_fn):
        return None

    if eval_type == "custom":
        metrics = pd.read_csv(eval_fn, delimiter="\t")
        # collapse metrics down to a single row
        ms: pd.DataFrame = metrics.set_index("set").stack().to_frame().T
        ms.columns = ["_".join(c) for c in ms.columns.values]
        return ms
    elif eval_type == "lightning":
        # this is old code for reading in the pytorch lightning metrics.txt instead of metrics_custom.txt
        metrics = pd.read_csv(eval_fn, delimiter=",", index_col=0, header=None).T
        # this is a quick hack to solve pandas reading in values as strings instead of floats
        for col in metrics.columns:
            if "pearson" in col or "spearman" in col:
                metrics[col] = metrics[col].astype(float)
        return metrics
    else:
        raise ValueError("unknown eval_type {}".format(eval_type))


def load_metrics_and_args(log_dirs: Union[str, list[str]],
                          eval_fn: str = "metrics_custom.txt",
                          eval_type: str = "custom",
                          args_fn: str = "args.txt",
                          args_type: str = "custom"):

    # if we're passed in a single log directory as a string, convert it to a list for consistent handling
    if isinstance(log_dirs, str):
        log_dirs = [log_dirs]

    rows = []
    for log_dir in log_dirs:
        m = load_metrics(log_dir, eval_fn=eval_fn, eval_type=eval_type)
        args = load_args_df(log_dir, args_fn=args_fn, args_type=args_type)

        if m is None:
            print("Unable to load metrics for {}, skipping...".format(basename(log_dir)))
        elif args is None:
            print("Unable to load args for {} skipping...".format(basename(log_dir)))
        else:
            # concat the args and metrics (horizontally)
            row = pd.concat((args.reset_index(drop=True), m.reset_index(drop=True)), axis=1, ignore_index=False)
            rows.append(row)

    if len(rows) == 0:
        print("Unable to load any metrics/args, returning None")
        return None
    elif len(rows) == 1:
        return rows[0]
    else:
        return pd.concat(rows, axis=0).reset_index(drop=True)


def check_for_failed_jobs(run_dir, eval_fn="metrics_custom.txt"):
    """ checks for failed jobs in the given HTCondor run directory """

    if not isdir(run_dir):
        raise FileNotFoundError("run_dir is not a directory: {}".format(run_dir))

    # get the number of expected jobs for this run from the args
    # can try env_vars.txt, args directory, args.tar.gz, or htcondor.sub
    args_dir = join(run_dir, "args")
    if isdir(args_dir):
        num_expected_jobs = len(os.listdir(args_dir))
    elif isfile(join(run_dir, "args.tar.gz")):
        # untarring not implemented
        raise FileNotFoundError("did not find args directory {}, but found args.tar.gz".format(args_dir))
    elif isfile(join(run_dir, "env_vars.txt")):
        raise NotImplementedError("did not find args dir or args.tar.gz, but found env_vars.txt which might contain"
                                  " the number of expected jobs. however, parsing it is not implemented")
    else:
        raise FileNotFoundError("did not find args directory, args.tar.gz, or env_vars.txt, unable to determine "
                                "number of expected jobs")

    # check if the training logs directory exists
    training_logs_dir = join(run_dir, "run_output", "training_logs")
    if not isdir(training_logs_dir):
        # old nn4dms code used "output" instead of "run_output", but there was a conflict that required changing dir
        raise FileNotFoundError("training logs directory does not exist: {}".format(training_logs_dir))

    # get the log directories for each job
    log_dirs = [join(training_logs_dir, x) for x in os.listdir(training_logs_dir) if isdir(join(training_logs_dir, x))]

    # there are TWO TYPES of failed jobs:
    # - Jobs that have a log directory but no final_evaluation.txt
    # - Jobs that don't even have a log directory
    all_job_ids = []
    failed_job_ids = []

    # jobs that have a log_dir but no final_evaluation.txt
    for ld in log_dirs:
        job_id = parse_log_dir_name(ld)["process"]
        all_job_ids.append(job_id)
        full_eval_fn = join(ld, eval_fn)
        if not isfile(full_eval_fn):
            failed_job_ids.append(job_id)

    # jobs that don't even have a log_dir
    no_log_dir_ids = list(set(map(str, range(0, num_expected_jobs))).difference(all_job_ids))

    # return the IDs of the failed jobs
    return failed_job_ids + no_log_dir_ids


def copy_failed_args(failed_job_ids, source_args_dir, out_args_dir):
    # todo: this function hasn't been tested
    # copy over failed job args
    copied_arg_ids = []
    skipped_arg_ids = []
    for fjid in failed_job_ids:
        out_fn = join(out_args_dir, "{}.txt".format(fjid))
        if not isfile(out_fn):
            shutil.copy(join(source_args_dir, "{}.txt".format(fjid)), out_fn)
            copied_arg_ids.append(fjid)
        else:
            skipped_arg_ids.append(fjid)
    return copied_arg_ids, skipped_arg_ids


def get_log_dirs(condor_run_dirs):
    """ get log directories for given condor run directories (can pass in single run dir or a list of them) """
    if not isinstance(condor_run_dirs, (list, tuple)):
        condor_run_dirs = [condor_run_dirs]
    log_dirs = []
    for rd in condor_run_dirs:
        log_dirs += [join(rd, "run_output/training_logs", x) for x in os.listdir(join(rd, "run_output/training_logs"))
                     if isdir(join(rd, "run_output/training_logs", x))]
    return log_dirs


def check_for_failed_jobs_uuid(run_dir, eval_fn="metrics_custom.txt", return_df=False) -> Union[pd.DataFrame, list]:
    """ checks for failed jobs in the given HTCondor run directory """

    if not isdir(run_dir):
        raise FileNotFoundError("run_dir is not a directory: {}".format(run_dir))

    # load the queue for this run which contains all the UUIDs and arg file numbers for this run
    queue_fn = join(run_dir, "queue.txt")

    # if there are additional queue files named queue_1, queue_2, etc.
    # then some failed args/UUIDs may have been run and failed multiple times
    # when we return condor_job_nums it would be useful to know all the failed job numbers
    # this doesn't affect generating new queues since those are generated from UUIDs
    # actually, let's use arg nums instead of UUIDs to do the mapping because future
    # frameworks might not use UUIDs
    queue_fns = [x for x in os.listdir(run_dir) if x.startswith("queue") and x.endswith(".txt")]
    queue_fns = ["queue_0.txt" if x == "queue.txt" else x for x in queue_fns]
    queue_fns = sorted(queue_fns, key=lambda x: int(x.split("_")[1].split(".")[0]))
    queue_fns = [join(run_dir, x) for x in queue_fns]
    # now create a mapping from arg_num to queue+job_num
    arg_num_to_queue_job_num = {}
    for queue_fn in queue_fns:
        if queue_fn.endswith("queue_0.txt"):
            # a bit of hackery to convert queue_0 back to queue.txt
            queue_fn = join(run_dir, "queue.txt")
        lines = utils.load_lines(queue_fn)
        for condor_job_num, l in enumerate(lines):
            a = l.split(",")[0]
            for arg_num in a.split("|"):
                # this overwrites the previous value, which means
                # if a job failed multiple times, we'll only get the last queue file
                arg_num_to_queue_job_num[arg_num] = (queue_fn, condor_job_num)

    # the main loop through the queue file
    lines = utils.load_lines(queue_fn)
    arg_nums = []
    uuids = []
    condor_job_nums = []
    for line in lines:
        a = line.split(",")[0]
        u = line.split(",")[1]
        for arg_num in a.split("|"):
            arg_nums.append(arg_num)
            # also want to get the condor job number
            queue_fn, condor_job_num = arg_num_to_queue_job_num[arg_num]
            condor_job_nums.append("{}:{}".format(basename(queue_fn)[:-4], condor_job_num))
        for uuid in u.split("|"):
            uuids.append(uuid)
        # for _ in range(len(a.split("|"))):
        #     condor_job_nums.append(condor_job_num)

    # queue = pd.read_csv(queue_fn, header=None, names=["arg_num", "uuid"])
    queue = pd.DataFrame({"arg_num": arg_nums, "uuid": uuids, "job_num": condor_job_nums})
    num_expected_jobs = len(queue)

    # check if the training logs directory exists
    training_logs_dir = join(run_dir, "run_output", "training_logs")
    if not isdir(training_logs_dir):
        # old nn4dms code used "output" instead of "run_output", but there was a conflict that required changing dir
        raise FileNotFoundError("training logs directory does not exist: {}".format(training_logs_dir))

    # get the log directories in the training_logs folder
    log_dirs = [join(training_logs_dir, x) for x in os.listdir(training_logs_dir) if isdir(join(training_logs_dir, x))]

    # there are TWO TYPES of failed jobs:
    # 1. Jobs that have a log directory but no eval_fn
    # 2. Jobs that don't even have a log directory
    # 2 shouldn't really happen anymore because log directories are now created ahead of time
    # still check for it though, for robustness, in case that changes

    # jobs that have a log_dir but no eval_fn
    no_eval_fn_uuids = []
    for ld in log_dirs:
        # todo: it'd be better to get the UUID from the args file
        uuid = utils.parse_log_dir_name(ld)["uuid"]
        if not isfile(join(ld, eval_fn)):
            no_eval_fn_uuids.append(uuid)

    # jobs that don't even have a log_dir
    found_uuids = [utils.parse_log_dir_name(ld)["uuid"] for ld in log_dirs]
    no_log_dir_uuids = list(set(queue["uuid"].to_list()).difference(found_uuids))

    if return_df:
        return queue[queue.uuid.isin(no_eval_fn_uuids + no_log_dir_uuids)]
    else:
        return no_eval_fn_uuids + no_log_dir_uuids


def create_new_queue(run_dir):
    """ creates a new queue.txt for failed jobs so they can be easily resubmitted """

    # get a dataframe of failed UUIDs
    failed_uuids = check_for_failed_jobs_uuid(run_dir, return_df=True)

    if len(failed_uuids) == 0:
        print("no failed UUIDs for run_dir: {}".format(run_dir))
    else:
        existing_versions = []
        for queue_fn in [x for x in os.listdir(run_dir) if x.startswith("queue_") and x.endswith(".txt")]:
            existing_versions.append(int(queue_fn.split("_")[1][:-4]))

        if len(existing_versions) == 0:
            version = 1
        else:
            version = max(existing_versions) + 1

        queue_fn = join(run_dir, "queue_{}.txt".format(version))

        print("found {} failed UUIDs, saving to {}".format(len(failed_uuids), queue_fn))

        # save as a new queue (increment by 1)
        failed_uuids[["arg_num", "uuid"]].to_csv(queue_fn, index=False, header=False)


def get_extrapolation_type(split_dir):
    # if this isn't an extrapolation eval_type, just return None for the extrapolation type
    # todo: for now, "standard" is counted as an extrapolation type becuase it is plotted along with
    #   the other extrapolation types, but eventually it should be removed from that plot
    extrapolation_eval_types = ("mutation", "position", "score", "regime", "standard", "standard-singles")
    eval_type = get_eval_type(split_dir)
    if eval_type not in extrapolation_eval_types:
        return None

    # for now, these eval types are the same as the extrapolation types
    # although in the future, we might have different types of score extrapolation
    if eval_type in ("mutation", "position", "score", "standard", "standard-singles"):
        return eval_type

    if eval_type == "regime":
        # determine the training regimes, those will be used to identify the extrapolation type
        # the test regimes will not go into the regime type... for now.
        # this could become a problem in the future if we have different test regime
        train_regimes = basename(split_dir).split("_")[1][6:].split("-")
        extrap_type = "regime_tr{}".format("-".join(train_regimes))
        return extrap_type

    raise ValueError("not sure how to get extrapolation type for eval_type: {}".format(eval_type))


def get_eval_type(split_dir):
    eval_type = basename(split_dir).split("_")[0]
    return eval_type


def get_train_size(split_dir):
    eval_type = get_eval_type(split_dir)
    if eval_type in ["reduced", "resampled"]:
        return int(basename(split_dir).split("_")[1][2:])
    else:
        return -1


def get_split_rep_num(split_dir):
    eval_type = get_eval_type(split_dir)
    if eval_type in ["reduced", "resampled"]:
        return int(basename(split_dir).split("_")[-1])
    else:
        return -1


def augment_data(data):
    data["eval_type"] = data["split_dir"].apply(get_eval_type)
    data["extrap_type"] = data["split_dir"].apply(get_extrapolation_type)
    data["train_size"] = data["split_dir"].apply(get_train_size)
    data["reduced_rep_num"] = data["split_dir"].apply(get_split_rep_num)
    data["split_dir_base"] = data["split_dir"].apply(os.path.basename)
    if "best_model_path" in data.columns:
        data["best_model_epoch"] = data["best_model_path"].apply(lambda p: basename(p).split("-")[0].split("=")[-1])
    if "best_model_score" in data.columns:
        data["best_model_score"] = data["best_model_score"].apply(lambda b: float(b) if b != "None" else None)
    if "test_pearson" in data:
        data["test_pearson"] = data["test_pearson"].astype(float)
    if "test_spearman" in data:
        data["test_spearman"] = data["test_spearman"].astype(float)
    # data["learning_rate"] = data["learning_rate"].astype(float)
    if "learning_rate" in data:
        data["learning_rate"] = data["learning_rate"].astype(float)
    if "standardize_targets" in data:
        data["standardize_targets"] = data["standardize_targets"].apply(lambda s: False if np.isnan(s) else s)
    if "target_offset" in data:
        data["target_offset"] = data["target_offset"].astype(float)
    return data


def check_failed_uuid(run_dirs: list[str], eval_fn: str = "metrics_custom.txt"):
    for rd in run_dirs:
        failed_job_ids = check_for_failed_jobs_uuid(rd, eval_fn, return_df=True)
        print("Run: {}, Num failed jobs: {}".format(basename(rd), len(failed_job_ids)))
        for uuid, job_num in zip(failed_job_ids["uuid"], failed_job_ids["job_num"]):
            print("UUID: {}, Job: {}".format(uuid, job_num))

        # print("Run: {}, Num failed jobs: {}, IDs: {}, Job Nums: {}".format(
        #     basename(rd), len(failed_job_ids),
        #     failed_job_ids["uuid"].tolist(),
        #     failed_job_ids["job_num"].tolist()))


def selection(data, selection_constants, criteria, minimize=True):
    grouped = data.groupby(selection_constants)
    if minimize:
        selected_idxs = grouped[criteria].idxmin()
    else:
        selected_idxs = grouped[criteria].idxmax()

    # if all the criteria values are nan, we idxmin and idxmax will return nan as the index
    # we can't use nan as an index, so filter it out....
    # this will result in those combinations of constants missing data, which is fine
    selected_idxs = selected_idxs.dropna()

    selected_data = data.loc[selected_idxs]
    return selected_idxs, selected_data


def process_run(run_dirs: list[str],
                eval_fn: str = "metrics_custom.txt",
                eval_type: str = "auto",
                args_fn: str = "args.txt",
                args_type: str = "auto",
                # whether to load the actual predictions and compute supplementary metrics
                compute_additional_metrics: bool = True,
                plot_name_fn: Optional[Callable] = None,
                perform_selection: bool = False,
                selection_criteria: str = "val_mse",
                selection_minimize: bool = True,
                selection_constants: Optional[str] = None):

    if eval_type == "auto":
        if basename(eval_fn) == "metrics.txt":
            eval_type = "lightning"
        elif basename(eval_fn) == "metrics_custom.txt":
            eval_type = "custom"
        else:
            raise ValueError("not sure how to determine eval_type from eval_fn: {}".format(eval_fn))

    if args_type == "auto":
        if basename(args_fn) == "args.txt":
            args_type = "custom"
        elif basename(args_fn) == "hparams.yaml":
            args_type = "lightning"
        else:
            raise ValueError("not sure how to determine args_type from args_fn: {}".format(args_fn))

    check_failed_uuid(run_dirs, eval_fn)
    log_dirs = get_log_dirs(run_dirs)
    run = load_metrics_and_args(log_dirs, eval_fn, eval_type, args_fn, args_type)
    run = augment_data(run)

    # compute additional metrics
    if compute_additional_metrics:
        # for each row in the "run" dataframe, we need to load the raw predictions, and compute
        # any supplementary metrics, and append them to the dataframe

        # preload the dataset dataframes for all datasets in the run
        # this should hopefully speed things up a bit
        ds_names = run["ds_name"].unique()
        ds_dfs = {}
        epistasis_thresholds = {}
        pos_epistasis_thresholds = {}
        neg_epistasis_thresholds = {}
        for ds_name in ds_names:
            ds_dfs[ds_name] = utils.load_dataset(ds_name)
            ds_dfs[ds_name]["abs_epistasis"] = np.abs(ds_dfs[ds_name]["epistasis"])

            # pre-compute epistasis thresholds for all datasets
            # this should also speed things up a bit
            # determine the epistasis thresholds for this dataset (90th percentile)
            # this computes the epistasis threshold using the whole dataset
            epistasis_threshold = ds_dfs[ds_name]["abs_epistasis"].quantile(0.90)
            epistasis_thresholds[ds_name] = epistasis_threshold
            pos_epistasis_threshold = ds_dfs[ds_name]["epistasis"].quantile(0.90)
            pos_epistasis_thresholds[ds_name] = pos_epistasis_threshold
            neg_epistasis_threshold = ds_dfs[ds_name]["epistasis"].quantile(0.10)
            neg_epistasis_thresholds[ds_name] = neg_epistasis_threshold

        # todo: store the additional metrics to update the dataframe all at once at the end
        addtl_metrics = []
        for index, mrow in tqdm(run.iterrows(), total=len(run)):
            ds_name = mrow["ds_name"]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=scipy.stats.ConstantInputWarning)
                additional_metrics = compute_epistasis_metrics(mrow["log_dir"],
                                                               epistasis_thresholds[ds_name],
                                                               pos_epistasis_thresholds[ds_name],
                                                               neg_epistasis_thresholds[ds_name],
                                                               ds=ds_dfs[ds_name])
            additional_metrics['index'] = index
            addtl_metrics.append(additional_metrics)
            # run.loc[index, additional_metrics.keys()] = additional_metrics.values()

        # update the dataframe all at once
        addtl_metrics_df = pd.DataFrame(addtl_metrics)
        addtl_metrics_df.set_index('index', inplace=True)

        run = run.merge(addtl_metrics_df, left_index=True, right_index=True)

    if perform_selection:
        selected_idxs, run = selection(data=run,
                                       selection_constants=selection_constants,
                                       criteria=selection_criteria,
                                       minimize=selection_minimize)

    if plot_name_fn is not None:
        run["plot_name"] = run.apply(plot_name_fn, axis=1)
    else:
        run["plot_name"] = None

    # fill in NaNs with 0 for spearman and pearson
    for col in [c for c in run.columns if "pearsonr" in c or "spearmanr" in c]:
        run[col] = run[col].fillna(0)

    return run


def compute_epistasis_metrics(log_dir: str,
                              epistasis_threshold: float,
                              pos_epistasis_threshold: float,
                              neg_epistasis_threshold: float,
                              ds: Optional[pd.DataFrame] = None) -> dict[str, float]:
    """ right now this only computes the epistasis metrics, but we could add more here """

    pred_df, set_names = load_predictions(log_dir, ds=ds, return_set_names=True)

    if "epistasis" not in pred_df.columns:
        raise ValueError("epistasis not in pred_df columns")

    if "abs_epistasis" not in pred_df.columns:
        pred_df["abs_epistasis"] = np.abs(pred_df["epistasis"])

    # compute spearmanr and pearsonr for the absolute, positive, and negative epistasis
    metrics = {}
    for metric in ["ep", "pos_ep", "neg_ep"]:
        if metric == "ep":
            filt_pred_df = pred_df[pred_df["abs_epistasis"] >= epistasis_threshold]
        elif metric == "pos_ep":
            filt_pred_df = pred_df[pred_df["epistasis"] >= pos_epistasis_threshold]
        elif metric == "neg_ep":
            filt_pred_df = pred_df[pred_df["epistasis"] <= neg_epistasis_threshold]
        else:
            raise ValueError("invalid metric: {}".format(metric))

        for set_name in set_names:
            filt_set_pred_df = filt_pred_df[filt_pred_df["set_name"] == set_name]

            # to handle "stest"... display as "test"
            if set_name == "stest":
                set_name = "test"

            # if this set is empty, that means there are no high epistasis variants in this set
            # also if there are less than 2 variants, we can't compute spearmanr or pearsonr
            # use NaN for the metric
            if len(filt_set_pred_df) < 2:
                metrics["{}_{}_spearmanr".format(metric, set_name)] = np.nan
                metrics["{}_{}_pearsonr".format(metric, set_name)] = np.nan

            else:
                # compute spearmanr and pearsonr for this set
                spearman_corr = scipy.stats.spearmanr(filt_set_pred_df["score"], filt_set_pred_df["prediction"])[0]
                metrics["{}_{}_spearmanr".format(metric, set_name)] = spearman_corr

                pearson_corr = scipy.stats.pearsonr(filt_set_pred_df["score"], filt_set_pred_df["prediction"])[0]
                metrics["{}_{}_pearsonr".format(metric, set_name)] = pearson_corr

    return metrics


def get_pos_encoding(uuid):
    """ get pos encoding based on the uuid """

    # load the uuid ident map
    uuid_map = pd.read_csv("output/model_index/uuid_ident_map.csv", index_col=0)
    if uuid in uuid_map.index:
        return uuid_map.loc[uuid]["pos_encoding"]
    else:
        raise ValueError("Unknown positional encoding for uuid {}".format(uuid))


def get_model_info(uuid, info):
    """ convenience function for accessing the uuid index map containing useful model info """

    # load the uuid ident map
    uuid_map = pd.read_csv("output/model_index/uuid_ident_map.csv", index_col=0)

    # make sure the uuid is in the map
    if uuid not in uuid_map.index:
        raise ValueError("Unknown uuid {}".format(uuid))

    if info not in uuid_map.columns:
        raise ValueError("Unknown info {}".format(info))

    return uuid_map.loc[uuid][info]


def get_median_log_dirs(reduced_df, plot_names):
    """ get log directories of median models for each plot name and train size"""
    reduced_df = reduced_df[reduced_df["plot_name"].isin(plot_names)]

    df_sorted = reduced_df.sort_values(by=['plot_name', 'train_size', 'test_spearmanr'])

    # group by 'plot_name' and 'train_size', and get the index of the median row for each group
    median_indices = df_sorted.groupby(['plot_name', 'train_size']).apply(lambda x: x.index[int(len(x) / 2)])

    # flatten the index and convert it to a list
    median_indices_list = np.array(median_indices).flatten().tolist()

    # get the rows corresponding to the median indices
    result = df_sorted.loc[median_indices_list]

    # print(result[["uuid", "plot_name", "train_size", "test_spearmanr"]])

    return result


def compute_epistasis(df):

    variant_dict = df.set_index('variant')['score'].to_dict()

    def compute_epistasis_score(variant):
        mutations = variant.split(",")
        if len(mutations) < 2:
            return np.nan
        try:
            single_mutation_scores = [variant_dict[m] for m in mutations]
            epistasis = variant_dict[variant] - sum(single_mutation_scores)
            return epistasis
        except KeyError:
            return np.nan

    return np.array([compute_epistasis_score(variant) for variant in variant_dict.keys()])


def main():

    # just a test for profiling compute_additional_metrics
    run_dirs = ["output/htcondor_runs/target/main_gb1/local_2023-02-23_15-54-32_main_gb1_ridge_oh"]

    def get_plot_name(row):
        return "ridge_sk" if row["model_name"] == "ridge" else "linear_sk"

    # sk = process_run(run_dirs, plot_name_fn=get_plot_name, compute_additional_metrics=True)

    log_dirs = get_log_dirs(run_dirs)
    ds = utils.load_dataset(ds_name="gb1")
    for log_dir in log_dirs:
        load_predictions(log_dir, ds=ds)


if __name__ == "__main__":
    main()
