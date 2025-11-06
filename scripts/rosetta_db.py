import pathlib
import tarfile

import sqlite3
import yaml
import tqdm

import numpy as np
import pandas as pd

ROSETTA_SCRATCH_DIR="../data/scratch/rosetta"

def get_sqlite_dbcon(parent="WT"):
    """ parent can be WT or 2D """
    return sqlite3.connect(get_sqlite_db(parent))

def get_sqlite_db(parent):
    """ parent can be WT or 2D """
    return pathlib.Path(ROSETTA_SCRATCH_DIR) / (parent + ".db")

def get_splits_dir(parent, mkdir=True):
    if parent not in ["2D", "WT"]:
        raise ValueError("parent must be 2D or WT. Got : " + parent)
    retdir = pathlib.Path(ROSETTA_SCRATCH_DIR) / (parent + "_splits")
    if mkdir:
        retdir.mkdir(parents=False, exist_ok=True)
    return retdir

def read_rosetta_score_file(fh):
    ret = pd.read_csv(fh, delim_whitespace=True, skiprows=1)
    del ret["SCORE:"] # the first column is "SCORE:" we do not want
    return ret

def load_chtc_results_to_sqlite(tar_gz_filelist, db_con, 
                                    table_name_postfix=""):
    for archive_file in tqdm.tqdm(tar_gz_filelist):
        tf = tarfile.open(archive_file, mode='r:gz')
        variant_info = yaml.safe_load(tf.extractfile("./info.yaml"))
        relaxed_score_df = read_rosetta_score_file(
                                tf.extractfile("./variant_relaxed_score.sc"))
        relaxed_score_df["variant"] = variant_info["Variant"]

        # only put the last score into the relax_scores table
        # the first score is just the mutation without any relaxation
        # the second (or the last) score is the relaxed mutation
        relaxed_score_df.iloc[-1:].to_sql(
                name="relaxed_scores" + table_name_postfix, con=db_con, 
                if_exists='append', index=False,
                dtype={"variant":'text', "description":'text'} )

        # Put the docked scores into the docked_scores table
        docked_score_df = read_rosetta_score_file(
                tf.extractfile("./variant_docked_score.sc"))
        docked_score_df["variant"] = variant_info["Variant"]
        docked_score_df.to_sql(
                name="docked_scores" + table_name_postfix, con=db_con, 
                if_exists='append', index=False,
                dtype={"variant":'text', "description":'text'})

def get_energy_scores_from_sqlite(db_con, 
                                  remove_description=True,
                                  table_name_postfix=""):
    """ Get an inner join of relaxed scores and docking scores

        remove_description : remove columns r_description and d_description
            which specify which model was selected by Rosetta out of the 
            hundreds of models that it built (nstructs). All remaining scores
            should be floats. 

    """
    df = pd.read_sql(
            f"select r.*, d.* from relaxed_scores{table_name_postfix} as r "
            f"inner join docked_scores{table_name_postfix} as d "
            f"on d.variant = r.variant;"
            , db_con)

    # We have duplicated column names because they have similar names in the
    # two columns now relabel the columns so that the first columns are
    # prepended with r_ for relaxation and the second set of columns are
    # prepended with d_ for docking

    # description should occur twice in the database columns. Find out where
    desciption_idxs = np.where(df.columns == "description")[0]
    assert(len(desciption_idxs) == 2)
    table_split = desciption_idxs[0] + 1

    df.columns = ["r_" + c for c in df.columns[:table_split]] \
                 + ["d_" + c for c in df.columns[table_split:-1]] \
                 + [df.columns[-1]] # this is "variant", the key for joining 
    
    columns_to_drop = ["d_variant"]

    if remove_description: # which structure was selected. 
        columns_to_drop += ["r_description","d_description"]

    df.drop(columns=columns_to_drop, inplace=True)

    return df

def get_energy_scores(parent, table_name_postfix=""):
    db_con = get_sqlite_dbcon(parent=parent)
    df = get_energy_scores_from_sqlite(db_con=db_con, 
            remove_description=True,
            table_name_postfix=table_name_postfix)
    db_con.close()
    return df






if __name__ == "__main__":
    # put in the 2D simulated scores from CHTC into sqlite
    #db_con = get_sqlite_dbcon(parent="2D")
    #chtc_results_dir=pathlib.Path("../data/scratch/rosetta/2D_triple_mutants")
    #load_chtc_results_to_sqlite(list(chtc_results_dir.glob("*.tar.gz")), 
    #               db_con=db_con)

    # put the test set data (which was done a second time) into table names
    # with a postfix
    db_con = get_sqlite_dbcon(parent="2D")
    chtc_results_dir=pathlib.Path(
            "../data/scratch/rosetta/2D_triple_mutants_test_set")
    load_chtc_results_to_sqlite(list(chtc_results_dir.glob("*.tar.gz")), 
            db_con=db_con,
            table_name_postfix="_redo_test_set")
