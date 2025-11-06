""" Encodes data in different formats """

from os.path import join, isfile, dirname
import argparse
from typing import Optional, Union, Sequence, Literal

import numpy as np
import pandas as pd

import rosetta_data as rd
import constants
import utils


def is_seq_level_encoding(encoding):
    seq_level_encodings = [
        "rosetta",
        "rosetta_total_score",
        
        "metl-l-2m-1d",
        "metl-l-2m-3d",

        "metl-l-2m-1d_total_score",
        "metl-l-2m-3d_total_score",

        "metl-l-2m-1d_bc1",
        "metl-l-2m-3d_bc1",

        "metl-g-20m-1d",
        "metl-g-20m-3d",
        "metl-g-50m-1d",
        "metl-g-50m-3d",

        "metl-g-20m-1d_total_score",
        "metl-g-20m-3d_total_score",
        "metl-g-50m-1d_total_score",
        "metl-g-50m-3d_total_score",

        "metl-g-20m-1d_bc2",
        "metl-g-20m-3d_bc2",
        "metl-g-50m-1d_bc2",
        "metl-g-50m-3d_bc2"

        "esm-8m", "esm-35m", "esm-150m",
    ]

    if encoding.lower() in seq_level_encodings:
        return True
    else:
        return False


def is_rosetta_encoding(encoding):
    """ check if the encoding is a Rosetta-based encoding
        or a combo of a Rosetta-based encoding and a non-Rosetta-based encoding """
    rosetta_encodings = ["rosetta", "rosetta_total_score"]

    # for purposes of standardizing the encoding, we treat METL-G as a Rosetta-based encoding
    metl_encodings = ["metl-g-20m-1d",
                      "metl-g-20m-3d",
                      "metl-g-50m-1d",
                      "metl-g-50m-3d",
                      "metl-l-2m-1d",
                      "metl-l-2m-3d"]
    # also append the total_score versions of the metl encodings
    metl_encodings += [enc + "_total_score" for enc in metl_encodings]
    rosetta_encodings += metl_encodings

    # # for testing... add esm encodings...
    # esm_encodings = ["esm-8m", "esm-35m", "esm-150m"]
    # rosetta_encodings += esm_encodings
    #
    # # for testing... add metl-g_bc2 encodings...
    # metl_encodings = ["metl-g-20m-1d_bc2", "metl-g-20m-3d_bc2", "metl-g-50m-1d_bc2", "metl-g-50m-3d_bc2"]
    # rosetta_encodings += metl_encodings

    for enc in encoding.split("+"):
        if enc in rosetta_encodings:
            return True
    return False


def get_rosetta_attributes():
    """ the rosetta attributes used in the gb1_rosetta encoding """

    all_attributes = constants.ROSETTA_ATTRIBUTES

    # exclude the same attributes that are excluded for source models
    # these are the default exclude_attributes set up in argparse for RosettaDataModule
    exclude_attributes = ['filter_total_score',
                          'dslf_fa13',
                          'res_count_all',
                          'linear_chainbreak',
                          'overlap_chainbreak']

    target_names = [attr for attr in all_attributes if attr not in exclude_attributes]

    return target_names


def encode_metl(metl_ident: str,
                ds_name: str,
                variants: Union[list[str], tuple[str]],
                indexing: Literal["0_indexed", "1_indexed"],
                backbone_cutoff: Optional[int] = None,
                target_names: Optional[Union[list[str], tuple[str]]] = None) -> np.ndarray:
    """ encode variants w/ METL-G """

    # target_names can only be used if backbone_cutoff is None
    if backbone_cutoff is not None and target_names is not None:
        raise ValueError("target_names can only be used if backbone_cutoff is None")

    if variants is not None and not (isinstance(variants, list) or isinstance(variants, tuple)):
        variants = [variants]

    # the key into constants.DATASETS for the METL-G numpy file
    if backbone_cutoff is not None:
        fn_key = "{}_bc{}_dms_cov_fn".format(metl_ident.lower(), backbone_cutoff)
    else:
        fn_key = "{}_dms_cov_fn".format(metl_ident.lower())

    if (fn_key not in constants.DATASETS[ds_name]) or (constants.DATASETS[ds_name][fn_key] is None):
        raise ValueError("No {} defined for dataset {} in constants".format(fn_key, ds_name))

    npy_fn = constants.DATASETS[ds_name][fn_key]

    # the METL-G dms coverage inference data is 0-indexed, so we need to make sure
    # the variants are also 0-indexed
    if indexing == "1_indexed":
        # convert these variants to 0-indexing for compatability with the DMS dataset
        variants = utils.convert_indexing(variants, offset=-1)

    # load from the METL-G inference data from file
    # the METL-G inference data contains METL-G Rosetta energy predictions for
    # every variant in the DMS dataset in the same order as the DMS dataset
    metl_data = np.load(npy_fn)

    # load the DMS data for this dataset because we need it to figure out what variants to
    # grab from the inference data
    df = utils.load_dataset(ds_name)

    # select the rows from metl_data corresponding to the variants we want
    # this approach ensures we get the data in the same order as the variant list
    temp = pd.DataFrame(data=metl_data, index=df["variant"])
    metl_data = temp.loc[variants].to_numpy()

    if backbone_cutoff is not None:
        return metl_data

    else:
        # now we need to get the correct columns from the METL-G energies
        # the target_names are the Rosetta energy names we want to use
        target_names = utils.get_rosetta_energy_names(target_names)

        # These are the default energy names you get when you call
        # get_rosetta_energy_names() without specifying any target_names
        metl_energy_names = utils.get_rosetta_energy_names()

        # get the indices of the target_names in the metl_energies
        # these are the columns we want to keep
        target_indices = [metl_energy_names.index(target_name) for target_name in target_names]

        # get the correct columns from the METL-G energies
        metl_data = metl_data[:, target_indices]

        return metl_data


def encode_rosetta(ds_name: str,
                   variants: Union[list[str], tuple[str]],
                   indexing: Literal["0_indexed", "1_indexed"] = "0_indexed",
                   target_names: Optional[Union[list[str], tuple[str]]] = None) -> np.ndarray:
    """ encode variants w/ Rosetta energies """

    if variants is not None and not (isinstance(variants, list) or isinstance(variants, tuple)):
        variants = [variants]

    if ("rosetta_dms_cov_fn" not in constants.DATASETS[ds_name]) or \
            (constants.DATASETS[ds_name]["rosetta_dms_cov_fn"] is None):
        raise ValueError("No 'rosetta_dms_cov_fn' defined for dataset {} in constants".format(ds_name))

    hdf_fn = constants.DATASETS[ds_name]["rosetta_dms_cov_fn"]

    if indexing == "0_indexed":
        # convert these variants to 1-indexing for compatability with the database
        variants = rd.convert_dms_to_rosettafy_indexing(ds_name, variants)

    elif indexing == "1_indexed":
        pass
    else:
        raise ValueError("Unknown indexing type")

    # load from hdf file (fastest, at least when encoding based on the dms_cov database)
    # might be slower if encoding from the full, 3gb+ database?
    data: pd.DataFrame = pd.read_hdf(hdf_fn)
    # use the default target_names_exclude (which should give us same energies as the source models)
    # that is, as long as I don't change RosettaDataModule argparse defaults :)
    target_names = utils.get_rosetta_energy_names(target_names)
    energies = data.set_index("mutations").loc[variants].reset_index()[target_names]

    # now convert the energies dataframe to be a numpy array
    energies = energies.to_numpy().astype(np.float32)

    # this dataset contains NaN for some variants. replace them with the mean value in the column.
    # https://stackoverflow.com/questions/69231756/how-to-fill-a-numpy-arrays-nan-values-with-the-means-of-their-columns
    col_mean = np.nanmean(energies, axis=0)
    nan_inds = np.where(np.isnan(energies))
    energies[nan_inds] = np.take(col_mean, nan_inds[1])

    return energies


def enc_one_hot(int_seqs: np.ndarray) -> np.ndarray:
    # enc = OneHotEncoder(categories=[range(constants.NUM_CHARS)] * int_seqs.shape[1], dtype=np.bool, sparse=False)
    # one_hot = enc.fit_transform(int_seqs).reshape((int_seqs.shape[0], int_seqs.shape[1], constants.NUM_CHARS))
    # NOTE: since this gets returned as float32 now, it will take up more space if saved to disk.
    # consider returning as boolean to save space on disk and just converting after it is loaded in.
    one_hot = np.eye(constants.NUM_CHARS)[int_seqs]
    return one_hot.astype(np.float32)


def enc_int_seqs_from_char_seqs(char_seqs, cls_token=False):
    cls_int = max(constants.C2I_MAPPING.values()) + 1  # integer for CLS token

    seq_ints = []
    for char_seq in char_seqs:
        if not cls_token:
            int_seq = [constants.C2I_MAPPING[c] for c in char_seq]
        else:
            int_seq = [cls_int] + [constants.C2I_MAPPING[c] for c in char_seq]
        seq_ints.append(int_seq)
    seq_ints = np.array(seq_ints)
    return seq_ints.astype(int)


def enc_int_seqs_from_variants(variants, wild_type_seq, wt_offset=0, cls_token=False):

    # handling CLS token
    enc_seq_len = len(wild_type_seq) if not cls_token else len(wild_type_seq) + 1
    cls_offset = 0 if not cls_token else 1
    cls_int = max(constants.C2I_MAPPING.values()) + 1  # integer for CLS token

    # convert wild type seq to integer encoding
    wild_type_int = np.zeros(enc_seq_len, dtype=np.uint8)
    if cls_token:
        wild_type_int[0] = cls_int
    for i, c in enumerate(wild_type_seq):
        wild_type_int[i + cls_offset] = constants.C2I_MAPPING[c]

    # tile the wild-type seq so
    seq_ints = np.tile(wild_type_int, (len(variants), 1))

    for i, variant in enumerate(variants):
        # special handling if we want to encode the wild-type seq
        # the seq_ints array is already filled with WT, so all we have to do is just ignore it
        # and it will be properly encoded
        if variant == "_wt":
            continue

        # variants are a list of mutations [mutation1, mutation2, ....]
        variant = variant.split(",")
        for mutation in variant:
            # mutations are in the form <original char><position><replacement char>
            position = int(mutation[1:-1])
            replacement = constants.C2I_MAPPING[mutation[-1]]
            seq_ints[i, position-wt_offset+cls_offset] = replacement

    return seq_ints.astype(int)


def encode_int_seqs(char_seqs: Optional[Union[list[str], tuple[str]]] = None,
                    variants: Optional[Union[list[str], tuple[str]]] = None,
                    wild_type_aa: Optional[str] = None,
                    wild_type_offset: Optional[int] = None,
                    cls_token: bool = False) -> np.ndarray:

    if char_seqs is None and variants is None:
        raise ValueError("Must provide either char_seqs or variants")

    elif variants is not None:
        if not isinstance(variants, list):
            variants = [variants]
        int_seqs = enc_int_seqs_from_variants(variants, wild_type_aa, wild_type_offset, cls_token)

    elif char_seqs is not None:
        if not isinstance(char_seqs, list):
            char_seqs = [char_seqs]
        int_seqs = enc_int_seqs_from_char_seqs(char_seqs, cls_token)

    return int_seqs


def enc_chars(int_seqs: np.ndarray) -> list[str]:
    """ encode as full character sequences """
    char_seqs = []
    for iseq in int_seqs:
        char_seq = "".join([constants.CHARS[i] for i in iseq])
        char_seqs.append(char_seq)
    return char_seqs


def encode_gb1_esm(variants, indexing="0_indexed"):
    """ loads the GB1 ESM representation as the sequence encoding """

    if indexing == "0_indexed":
        pass
    elif indexing == "1_indexed":
        # convert variants to 0-indexing for compatability with DMS data
        variants = utils.convert_indexing(variants, offset=-1)
    else:
        raise ValueError("Unknown indexing type")

    # load the gb1 dms dataset (ESM data is in order of variants)
    ds = utils.load_dataset("gb1")

    # load the encoding
    esm_data_fn = "output/htcondor_runs/condor_rtl_2022-02-15_12-07-08_gb1_esm_inference/" \
                  "output/esm_predictions/gb1/predictions.npy"
    esm_data = np.load(esm_data_fn)

    # join the dataframe, so that we can access the correct data with .loc
    # a (probably better) alternative is to just get the indices of the variants from the dms ds
    # and then slice directly into esm_data using the indices
    joined_df = pd.DataFrame(data=esm_data, index=ds["variant"])

    return joined_df.loc[variants].to_numpy()


def encode_esm(esm_ident: str,
               ds_name: str,
               variants: Union[list[str], tuple[str]],
               indexing: Literal["0_indexed", "1_indexed"]) -> np.ndarray:

    if variants is not None and not (isinstance(variants, list) or isinstance(variants, tuple)):
        variants = [variants]

    # the key into constants.DATASETS for the esm numpy file
    fn_key = "{}_dms_cov_fn".format(esm_ident.lower())

    if (fn_key not in constants.DATASETS[ds_name]) or (constants.DATASETS[ds_name][fn_key] is None):
        raise ValueError("No {} defined for dataset {} in constants".format(fn_key, ds_name))

    # load the ESM data
    npy_fn = constants.DATASETS[ds_name][fn_key]
    esm_data = np.load(npy_fn)

    # load the DMS data for this dataset
    df = utils.load_dataset(ds_name)

    if indexing == "1_indexed":
        # convert these variants to 0-indexing for compatability with the DMS dataset
        variants = utils.convert_indexing(variants, offset=-1)

    # select the rows from metl_data corresponding to the variants we want
    # this approach ensures we get the data in the same order as the variant list
    temp = pd.DataFrame(data=esm_data, index=df["variant"])
    esm_data = temp.loc[variants].to_numpy()

    return esm_data


def concat_encodings(encoding: str,
                     encoded_data: Sequence[np.ndarray],
                     squeeze_first_dim: bool = False) -> np.ndarray:

    encodings = encoding.split("+")

    # there are multiple ways to concatenate the encoded data
    # depending on whether we have a sequence-level or residue-level encoding
    # and whether we want to concatenate on the residue-level or the sequence-level
    # handle each case individually for now...

    # are any of the encodings sequence-level encodings?
    seq_level_encodings = [encoding for encoding in encodings if is_seq_level_encoding(encoding)]
    # rosetta_encodings = [encoding for encoding in encodings if is_rosetta_encoding(encoding)]

    if len(encoded_data) == 1:
        encoded_data = encoded_data[0]

    elif len(encoded_data) == 2 and "one_hot" in encodings and seq_level_encodings:
        # flatten the one_hot encoding residue-level encoding to concatenate
        # with the sequence-level rosetta encoding
        # get the indices of one_hot and Rosetta encoding in the encoded_data
        oh_idx = encodings.index("one_hot")
        r_idx = encodings.index(seq_level_encodings[0])

        # flatten the one_hot encoding
        oh_encoding = encoded_data[oh_idx].reshape(encoded_data[oh_idx].shape[0], -1)
        r_encoding = encoded_data[r_idx]

        # concatenate the encodings in the same order as the encodings list
        if oh_idx < r_idx:
            encoded_data = np.concatenate((oh_encoding, r_encoding), axis=1)
        else:
            encoded_data = np.concatenate((r_encoding, oh_encoding), axis=1)

    elif len(encoded_data) >= 2:
        raise ValueError("not sure how to concatenate the given encodings: {}".format(encodings))

    # squeeze the first dimension for singletons if requested
    if squeeze_first_dim and encoded_data.shape[0] == 1:
        encoded_data = encoded_data.squeeze(axis=0)

    return encoded_data


def encode(encoding: str,
           char_seqs: Optional[list[str]] = None,
           variants: Optional[list[str]] = None,
           ds_name: Optional[str] = None,
           wt_aa: Optional[str] = None,
           wt_offset: Optional[int] = None,
           indexing: str = "0_indexed",
           cls_token: bool = False,
           concat: bool = True):

    """ the main encoding function that will encode the given sequences or variants and return the encoded data """

    # error checking: rosetta encoding requires variants, not char seqs for now
    if is_rosetta_encoding(encoding) and variants is None:
        raise ValueError("Rosetta encoding requires variants, not char_seqs")
    if is_rosetta_encoding(encoding) and ds_name is None:
        raise ValueError("Need to specify ds_name for Rosetta encoding because we need to look up "
                         "the rosetta dms coverage data from constants")

    if variants is None and char_seqs is None:
        raise ValueError("must provide either variants or full sequences to encode")
    if variants is not None and ((ds_name is None) and ((wt_aa is None) or (wt_offset is None))):
        raise ValueError("if providing variants, must also provide (wt_aa and wt_offset) or "
                         "ds_name so I can look up the WT sequence myself")

    if cls_token and encoding != "int_seqs":
        raise ValueError("encoding currently only supports cls tokens for int_seqs encoding, got {}".format(encoding))

    # are we encoding a single variant or sequence? if so, keep track, so we can return the correct shape
    single = False
    if variants is not None and not (isinstance(variants, list) or isinstance(variants, tuple)):
        single = True
        variants = [variants]
    elif char_seqs is not None and not (isinstance(char_seqs, list) or isinstance(char_seqs, tuple)):
        single = True
        char_seqs = [char_seqs]

    # this function expects 0-based indexed variants (if variants are given)
    # if the given variants are 1-based, convert them to 0-based here
    if indexing == "0_indexed":
        pass
    elif indexing == "1_indexed":
        variants = utils.convert_indexing(variants, offset=-1)
    else:
        raise ValueError("unknown indexing type")

    if ds_name is not None:
        wt_aa = constants.DATASETS[ds_name]["wt_aa"]
        # if we specify a wt_offset with a ds_name, override the one from constants
        # todo: wt_offset override, need to handle database variants being 1-indexed vs. code expecting 0-indexed
        if wt_offset is None:
            wt_offset = constants.DATASETS[ds_name]["wt_ofs"]

    # convert given variants or char sequences to integer sequences
    # this may be a bit slower, but easier to program
    # this isn't required at all if we are just encoding Rosetta encoding, but
    # we are just going to deal with the overhead for now
    int_seqs = encode_int_seqs(char_seqs=char_seqs, variants=variants,
                               wild_type_aa=wt_aa, wild_type_offset=wt_offset, cls_token=cls_token)

    # encode variants using int seqs
    encoding = encoding.lower()
    encodings = encoding.split("+")
    encoded_data = []
    for enc in encodings:
        if enc == "one_hot":
            encoded_data.append(enc_one_hot(int_seqs))
        elif enc == "int_seqs":
            encoded_data.append(int_seqs)
        elif enc == "char_seqs":
            encoded_data.append(enc_chars(int_seqs))
        elif enc == "rosetta":
            # regardless of what indexing is passed into encode(), by this point it is 0-indexed
            encoded_data.append(encode_rosetta(ds_name, variants, indexing="0_indexed"))
        elif enc == "rosetta_total_score":
            encoded_data.append(encode_rosetta(ds_name, variants, indexing="0_indexed", target_names=["total_score"]))
        elif enc in ["metl-g-20m-1d", "metl-g-20m-3d",
                     "metl-g-50m-1d", "metl-g-50m-3d",
                     "metl-l-2m-1d", "metl-l-2m-3d"]:
            ed = encode_metl(enc, ds_name, variants, indexing="0_indexed")
            encoded_data.append(ed)

        elif enc in ["metl-g-20m-1d_bc2", "metl-g-20m-3d_bc2",
                     "metl-g-50m-1d_bc2", "metl-g-50m-3d_bc2",
                     "metl-l-2m-1d_bc1", "metl-l-2m-3d_bc1"]:
            bc = int(enc.split("_")[-1][-1])
            ed = encode_metl(enc.split("_")[0], ds_name, variants, indexing="0_indexed", backbone_cutoff=bc)
            encoded_data.append(ed)

        elif enc in ["metl-g-20m-1d_total_score", "metl-g-20m-3d_total_score",
                     "metl-g-50m-1d_total_score", "metl-g-50m-3d_total_score",
                     "metl-l-2m-1d_total_score", "metl-l-2m-3d_total_score"]:
            ed = encode_metl(enc.split("_")[0], ds_name, variants, indexing="0_indexed", target_names=["total_score"])
            encoded_data.append(ed)

        elif enc in ["esm-8m", "esm-35m", "esm-150m"]:
            ed = encode_esm(enc, ds_name, variants, indexing="0_indexed")
            encoded_data.append(ed)
        else:
            raise ValueError("err: encountered unknown encoding: {}".format(enc))

    if len(encoded_data) == 1:
        encoded_data = encoded_data[0]
        if single:
            encoded_data = encoded_data[0]
    elif len(encoded_data) > 1 and concat:
        encoded_data = concat_encodings(encoding, encoded_data, squeeze_first_dim=single)

    return encoded_data


def encode_full_dataset(ds_name, encoding):
    # load the dataset
    ds = utils.load_dataset(ds_name=ds_name)
    # encode the data
    encoded_data = encode(encoding=encoding, variants=ds["variant"].tolist(), ds_name=ds_name)
    return encoded_data


def encode_full_dataset_and_save(ds_name, encoding):
    """ encoding a full dataset """
    out_fn = join(constants.DATASETS[ds_name]["ds_dir"], "enc_{}_{}.npy".format(ds_name, encoding))
    if isfile(out_fn):
        print("err: encoded data already exists: {}".format(out_fn))
        return
    encoded_data = encode_full_dataset(ds_name, encoding)
    np.save(out_fn, encoded_data)
    return encoded_data


def main(args):

    if args.ds_name == "all":
        ds_names = constants.DATASETS.keys()
    else:
        ds_names = [args.ds_name]

    if args.encoding == "all":
        encodings = ["one_hot"]
    else:
        encodings = [args.encoding]

    for ds_name in ds_names:
        for encoding in encodings:
            encode_full_dataset_and_save(ds_name, encoding)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ds_name",
                        help="name of the dataset",
                        type=str)
    parser.add_argument("encoding",
                        help="what encoding to use",
                        type=str,
                        choices=["one_hot", "all"])
    main(parser.parse_args())
