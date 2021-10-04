import gzip 

import Bio
import Bio.SeqIO

import numpy as np
import pandas as pd

datadir = "data/jared_PacBio_data_2"

WT_fasta = Bio.SeqIO.read(datadir + "/2D.fasta", format="fasta")
WT_DNA = WT_fasta.seq
WT = WT_DNA.translate()

start = 53
end = 875

length_cutoff = 1
min_phred_base = 93
mut_cut = 15

def hamming_dist(s1, s2):
    assert(len(s1) == len(s2))
    return sum(1 for (a, b) in zip(s1, s2) if a != b)


def complement_and_translate(r, translate=False):
    sel = r[start:end].seq
    sel_r = r.reverse_complement()[start:end].seq
    dist = hamming_dist(sel, WT_DNA)
    dist_r = hamming_dist(sel_r, WT_DNA)
    ret = None
    if dist_r > dist:
        ret = sel
    else:
        ret = sel_r
    if translate:
        ret = ret.translate()
    return ret

def get_fastq_records(filename):
    with gzip.open(filename, "rt") as fh:
        return list(Bio.SeqIO.parse(fh, "fastq"))

def length_filter(r):
    return len(r) >= length_cutoff*(end)

def quality_filter(r):
    return min(r.letter_annotations["phred_quality"][start:end], default=-1) \
                >= min_phred_base

def num_muts_filter(s):
    return hamming_dist(s, WT_DNA) < mut_cut

if __name__ == "__main__":
    import sys
    import argparse
    import logging
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--fastq_file",
        help="fastq input file", required=True)
    parser.add_argument("-o", "--output_filename",
        help="Output filename for filtered sequences", required=True)
    parser.add_argument("-l", "--loglevel",
        help="Logging Level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout, 
                            level=getattr(logging, args.loglevel))

    logging.info(f"Reading filename: {args.fastq_file}")
    records = get_fastq_records(args.fastq_file)
    logging.info(f"Number of records read                : {len(records)}")

    records = list(filter(length_filter, records))
    logging.info(f"Number of records after length filter : {len(records)}")

    #records = list(filter(quality_filter, records))
    #logging.info(f"Number of records after quality filter: {len(records)}")

    records = list(map(complement_and_translate, records))
    logging.info(f"Number of records after translate map : {len(records)}")

    records = list(filter(num_muts_filter, records))
    logging.info(f"Number of records after nummuts filter: {len(records)}")

    with open(args.output_filename, "wt")  as fh:
        for s in records:
            print(str(s), file=fh)

    logging.info(f"Number of records written             : {len(records)}")
    


