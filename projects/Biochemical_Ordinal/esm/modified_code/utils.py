""" general utility functions used throughput codebase """

import os
import uuid
from functools import reduce
from itertools import groupby
from os.path import join, isfile, isdir, basename
from typing import Optional, Union

import numpy as np
import pandas as pd
import shortuuid

import constants
import encode


def mkdir(d):
    """ creates given dir if it does not already exist """
    if not isdir(d):
        os.makedirs(d)


def load_lines(fn):
    """ loads each line from given file """
    lines = []
    with open(fn, "r") as f_handle:
        for line in f_handle:
            lines.append(line.strip())
    return lines


def save_lines(out_fn, lines):
    """ saves each line in data to given file """
    with open(out_fn, "w") as f_handle:
        for line in lines:
            f_handle.write("{}\n".format(line))


def load_dataset(ds_name: Optional[str] = None,
                 ds_fn: Optional[str] = None,
                 sort_mutations: bool = True,
                 load_epistasis: bool = True):

    """ load a dataset as pandas dataframe """
    if ds_name is None and ds_fn is None:
        raise ValueError("must provide either ds_name or ds_fn to load a dataset")

    if ds_fn is None:
        ds_fn = constants.DATASETS[ds_name]["ds_fn"]

    if not isfile(ds_fn):
        raise FileNotFoundError("can't load dataset, file doesn't exist: {}".format(ds_fn))

    ds = pd.read_csv(ds_fn, sep="\t")

    # ensure variants are in sorted order
    # all our datasets should have sorted mutations now
    # no need to do this anymore! default changed to soft_mutations = False
    if sort_mutations and "variant" in ds:
        ds["variant"] = sort_variant_mutations(ds["variant"])

    # load epistasis data if it exists
    epistasis_fn = ds_fn.replace(".tsv", "_epistasis.tsv")
    if load_epistasis and isfile(epistasis_fn):
        epistasis_df = pd.read_csv(epistasis_fn, sep="\t")
        # merge epistasis data with dataset
        ds = pd.merge(ds, epistasis_df, on="variant", how="left")

    return ds


def save_args(args_dict, out_fn, ignore=None):
    """ save argparse arguments dictionary back to a file that can be used as input to regression.py """

    with open(out_fn, "w") as f:
        for k, v in args_dict.items():
            # ignore these special arguments
            if (ignore is None) or (k not in ignore):
                # if a flag is set to false, dont include it in the argument file
                if (not isinstance(v, bool)) or (isinstance(v, bool) and v):
                    f.write("--{}\n".format(k))
                    # if a flag is true, no need to specify the "true" value
                    if not isinstance(v, bool):
                        if isinstance(v, list):
                            for lv in v:
                                f.write("{}\n".format(lv))
                        else:
                            f.write("{}\n".format(v))


def convert_indexing(variants, offset):
    """ convert between 0-indexed and 1-indexed """
    converted_to_list = False
    if not isinstance(variants, list) and not isinstance(variants, tuple) and not isinstance(variants, pd.Series):
        converted_to_list = True
        variants = [variants]

    converted = [",".join(["{}{}{}".format(mut[0], int(mut[1:-1]) + offset, mut[-1])
                           for mut in v.split(",")])
                 for v in variants]

    if converted_to_list:
        converted = converted[0]

    return converted


def gb1_seq2variant(seq):
    wt_seq = constants.DATASETS["gb1"]["wt_aa"]
    if len(seq) != len(wt_seq):
        raise ValueError("seq is not the same length as wt_seq")

    mut_positions = [i for i in range(len(wt_seq)) if wt_seq[i] != seq[i]]

    mut_strings = []
    for mut_pos in mut_positions:
        ms = "{}{}{}".format(wt_seq[mut_pos], mut_pos, seq[mut_pos])
        mut_strings.append(ms)

    return ",".join(mut_strings)


def sort_variant_mutations(variants):
    """ put variant mutations in sorted order by position """
    sorted_variants = []
    for variant in variants:
        muts = variant.split(",")
        positions = [int(mut[1:-1]) for mut in muts]
        # now sort muts by positions index, then join on "," to recreate variant
        sorted_muts = [x for x, _ in sorted(zip(muts, positions), key=lambda pair: pair[1])]
        sorted_variants.append(",".join(sorted_muts))
    return sorted_variants


def find_next_sequential_dirname(base_dir):
    """ finds the next sequential dirname by appending _i where i is the dir number """
    next_dir = base_dir
    if isdir(base_dir):
        i = 2
        next_dir = base_dir + "_{}".format(i)
        while isdir(next_dir):
            i += 1
            next_dir = base_dir + "_{}".format(i)
    return next_dir


def gen_model_uuid():
    my_uuid = shortuuid.encode(uuid.uuid4())[:8]
    return my_uuid


def all_equal(iterable):
    """ check if all list elements are equal (from itertools recipes)
        https://docs.python.org/3/library/itertools.html#itertools-recipes """
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def log_dir_name(log_dir_base, my_uuid):
    # maybe this should go in shared_model (or a new model_utils?) instead
    return join(log_dir_base, my_uuid)


def parse_log_dir_name(ld):
    return {"uuid": basename(ld)}


def sort_variants(variants):
    """ sort variants by number of mutations and mutation positions
        used to get a master order for sorting the dataframe """

    # first sort by number of mutations
    def key_func_1(v):
        return len(v.split(","))

    # next sort by positions of the mutations
    def key_func_2(v):
        return [int(x[1:-1]) for x in v.split(",")]

    # finally sort by the amino acid order defined in constants.CHARS
    def key_func_3(v):
        return [constants.CHARS.index(x[-1]) for x in v.split(",")]

    sv = sorted(variants, key=lambda v: [key_func_1(v), key_func_2(v), key_func_3(v)])

    return sv


def get_rosetta_energy_names(target_names: Optional[Union[list[str], tuple[str]]] = None,
                             target_names_exclude: Union[list[str], tuple[str]] = (
                             'filter_total_score', 'dslf_fa13', 'res_count_all', 'linear_chainbreak',
                             'overlap_chainbreak')):
    """ error checking and set up for target names """
    if target_names is None:
        return [attr for attr in constants.ROSETTA_ATTRIBUTES if attr not in target_names_exclude]
    elif isinstance(target_names, list) or isinstance(target_names, tuple):
        if all([tg in constants.ROSETTA_ATTRIBUTES for tg in target_names]):
            return target_names
        else:
            raise ValueError("some target_names not found in master list")
    else:
        raise ValueError("target_names should be a list: {}".format(target_names))


def get_module_by_name(module, access_string):
    """ https://discuss.pytorch.org/t/how-to-access-to-a-layer-by-module-name/83797/8 """
    names = access_string.split(sep='.')
    return reduce(getattr, names, module)