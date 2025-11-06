import gzip 

import numpy as np
import pandas as pd

import Bio
import Bio.SeqIO
import pysam

datadir = "../data/2D"

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
    parser.add_argument("-i", "--bam_filename",
        help="bam input file", required=True)
    parser.add_argument("-o", "--output_filename",
        help="Output filename for filtered sequences", required=True)
    parser.add_argument("-l", "--loglevel",
        help="Logging Level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout, 
                            level=getattr(logging, args.loglevel))

    logging.info(f"Reading filename: {args.bam_filename}")
    samfile = pysam.AlignmentFile(args.bam_filename, "rb", check_sq=False)

    with open(args.output_filename, "wt")  as fh:
        print("query_name\tnp\tlength\t{rc_better}"
              "ref_dist\tseq\tqual", file=fh)
        for i, line in enumerate(samfile):
            read_length = len(line.seq)
            seq = "" # snippet of sequence from start to end
            qual = ""
            rc_better = False
            ref_dist = -1
            np = -1
            try:
                np = line.get_tag('np')
            except KeyError:
                pass
            if read_length >= end:
                seq = line.seq[start:end]
                qual = line.qual[start:end]
                ref_dist = hamming_dist(seq, WT_DNA)
                if ref_dist > 100: # also check the reverse complement
                    r_seq = Bio.Seq.Seq(line.seq).reverse_complement()[start:end]
                    r_ref_dist = hamming_dist(r_seq, WT_DNA)
                    if r_ref_dist < ref_dist: # switch seqs to rev complement
                        rc_better = True
                        seq = r_seq
                        ref_dist = r_ref_dist
            print(f"{line.query_name}\t{np}\t{read_length}\t"
                  f"{rc_better}\t{ref_dist}\t{seq}\t{qual}", file=fh)
            print('\r' + str(i), end='')

    


