import pathlib

import tarfile
import pathlib

import sqlite3
import yaml
import tqdm

import pandas as pd

SQLITE3_DB_LOC="../data/scratch/rosetta" 

def get_sqlite_dbcon(parent="WT"):
    """ parent can be WT or 2D """
    return sqlite3.connect(get_sqlite_db(parent))

def get_sqlite_db(parent):
    """ parent can be WT or 2D """
    return SQLITE3_DB_LOC + "/" + parent + ".db"

def read_rosetta_score_file(fh):
    ret = pd.read_csv(fh, delim_whitespace=True, skiprows=1)
    del ret["SCORE:"] # the first column is "SCORE:" we do not want
    return ret

def load_chtc_results_to_sqlite(tar_gz_filelist, db_con):
    for archive_file in tqdm.tqdm(tar_gz_filelist):
        tf = tarfile.open(archive_file, mode='r:gz')
        variant_info = yaml.safe_load(tf.extractfile("./info.yaml"))
        relaxed_score_df = read_rosetta_score_file(
                                tf.extractfile("./variant_relaxed_score.sc"))
        relaxed_score_df["variant"] = variant_info["Variant"]

        # only put the last score into the relax_scores table
        # the first score is just the mutation without any relaxation
        # the second (or the last) score is the relaxed mutation
        relaxed_score_df.iloc[-1:].to_sql(name="relaxed_scores", con=db_con, 
                if_exists='append', index=False,
                dtype={"variant":'text', "description":'text'} )

        # Put the docked scores into the docked_scores table
        docked_score_df = read_rosetta_score_file(
                tf.extractfile("./variant_docked_score.sc"))
        docked_score_df["variant"] = variant_info["Variant"]
        docked_score_df.to_sql(name="docked_scores", con=db_con, 
                 if_exists='append', index=False,
                 dtype={"variant":'text', "description":'text'})


if __name__ == "__main__":
#    db_con = get_sqlite_dbcon(parent="WT")
#
#    chtc_results_dir=pathlib.Path("../data/scratch/rosetta/results_WT_campaign1")
#    load_chtc_results_to_sqlite(list(chtc_results_dir.glob("*.tar.gz")), db_con)
#
#    chtc_results_dir=pathlib.Path("../data/scratch/rosetta/results_WT_random")
#    load_chtc_results_to_sqlite(list(chtc_results_dir.glob("*.tar.gz")), db_con)
#

    db_con = get_sqlite_dbcon(parent="2D")

    chtc_results_dir=pathlib.Path("../data/scratch/rosetta/results_2D/")
    load_chtc_results_to_sqlite(list(chtc_results_dir.glob("*.tar.gz")), db_con)

