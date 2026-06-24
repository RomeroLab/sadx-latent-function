import pathlib
import numpy as np
import pandas as pd

import Bio
import Bio.Seq
import Bio.SeqIO

import pysam
from pysam import CSOFT_CLIP, CEQUAL, CDIFF


import matplotlib.pyplot as plt
import seaborn as sns


# the aligned bams are here (after processing with pbmm2)
samples_file=pathlib.Path("../data/scratch/r64120_20221013_213137/3_C01/00Samples")
data_dir = pathlib.Path(f"../data/scratch/Oct22")

# This is SadX without the MBP part (the OG wild-type)
SAD_WT_DNA = Bio.SeqIO.read("../data/SadA D157G.fasta", format="fasta").seq
SAD_WT_AA = SAD_WT_DNA.translate()

# Parents in each round (DNA sequences)
PARENT_1VH_DNA = Bio.SeqIO.read("../data/1-VH.fasta", format="fasta").seq
PARENT_2L_DNA = Bio.SeqIO.read("../data/2-L.fasta", format="fasta").seq
PARENT_3VRL_DNA = Bio.SeqIO.read("../data/3-VRL.fasta", format="fasta").seq

parent_d = {
        "1VH":PARENT_1VH_DNA,
        "2L":PARENT_2L_DNA,
        "3VRL":PARENT_3VRL_DNA
        }

# where we expect the parent DNA sequences to align to each read
# 
ref_start = 53
ref_end = 875
ref_len = ref_end - ref_start

# We should have atleast a few bases past the ends
min_length_past_ends = 1
max_mutants = 15
min_phred_base = 93

INTERNAL_ALLOWED_CIGAR_OPS = set([CDIFF, CEQUAL])


def hamming_dist(s1, s2):
    assert(len(s1) == len(s2))
    return sum(1 for (a, b) in zip(s1, s2) if a != b)


def shorten_seq(s, wt):
    assert(len(s) == len(wt)) # don't do deletions for now
    ret = ",".join(f"{wi}{i+1}{si}" for i, (si, wi) 
                   in enumerate(zip(s, wt)) if wi != si)
    if not len(ret): # then we have wild-type
        ret = f"*0*" # special return for wild-type
    return ret
    

num_samples_placed_d = {
        "1VH_H":4,
        "1VH_P":2,
        "1VH_L":349,
        "1VH_N":454,
        "2L_H":29,
        "2L_P":52,
        "2L_L":532,
        "2L_N":287,
        "3VRL_H":81,
        "3VRL_P":222,
        "3VRL_L":266,
        "3VRL_N":330 
}

if __name__ == "__main__":
    TESTING = False

    output_dir = pathlib.Path("../data/scratch/Oct22")
    samples = pd.read_csv(samples_file, sep="\t", skiprows=1)

    # write out sample counts to this file
    samples_counts = output_dir / "Samples.csv"
    with open(samples_counts, "wt") as fh_out:
        print("barcode,sample,num_placed,num_seqs,num_kept,"
              "align_filter,length_filter,quality_filter", 
              file=fh_out)
 

    for b_idx, barcode in enumerate(samples.Barcode):
        bamfile = data_dir / f"{barcode}.bam"

        bio_sample = samples.Bio_Sample[b_idx]
        parent = bio_sample[:bio_sample.find("_")]
        print(f"Processing : {bamfile}, {bio_sample} {parent}")

        wt_dna = parent_d[parent]
        wt = wt_dna.translate()

        keep_seqs = []

        # open alignment file
        save = pysam.set_verbosity(0)
        samfile = pysam.AlignmentFile(bamfile, "rb", check_header=False,
                                        check_sq=False)
        pysam.set_verbosity(save)

        align_filter_counter = 0
        length_filter_counter = 0
        quality_filter_counter = 0
        for i, line in enumerate(samfile):
            ct = line.cigartuples
            if ((ct[0][0] != CSOFT_CLIP) and (ct[-1][0] != CSOFT_CLIP)):
                continue  # the ends of the cigar string are not soft clipped
            start = ct[0][1]
            if ((start < min_length_past_ends) 
                    or (ct[-1][1] < min_length_past_ends)):
                # we should have atleast min_length_past_ends bases at the
                # start and the end of this read
                continue 
            if (any(x not in INTERNAL_ALLOWED_CIGAR_OPS for x, _ in ct[1:-1])):
                # make sure the internal cigar ops are allowed
                # indels are removed here
                continue
            align_filter_counter += 1 # we passed all the aligned filters

            # read length between end clips
            read_len = sum(x for _, x in ct[1:-1])
            if read_len != ref_len: # length filter
                continue
            length_filter_counter += 1

            # now check the quality scores of the section that matches
            # the reference
            end = start + read_len
            qual = np.array(line.get_forward_qualities()[start:end])
            if np.any(qual < min_phred_base): # quality filter
                continue
            quality_filter_counter += 1

            seq = line.seq[start:end] 
            ref_dist = hamming_dist(seq, wt_dna)
            ss_dna = shorten_seq(seq, wt_dna)
            dna_dist = hamming_dist(seq, wt_dna)

            seq_translate = Bio.Seq.Seq(seq).translate()
            ss_aa = shorten_seq(seq_translate, wt)
            aa_dist = hamming_dist(seq_translate, wt) 

            keep_seqs.append([ss_dna, dna_dist, ss_aa, aa_dist, 
                            ref_dist])
            if TESTING and (i > 1000):
                break
        #break
        samfile.close()
        keep_seqs = pd.DataFrame(keep_seqs, 
                    columns=["ss_dna", "dna_dist", "ss_aa", "aa_dist",
                             "ref_dist"])     
        keep_seqs.to_csv(output_dir / f"{barcode}.tsv", sep="\t", index=False)
        num_samples_placed = num_samples_placed_d[bio_sample]
        with open(samples_counts, "a") as fh_out:
            print(f"{barcode},{bio_sample},{num_samples_placed},"
                  f"{i},{len(keep_seqs)},{align_filter_counter},"
                  f"{length_filter_counter},{quality_filter_counter}",
                  file=fh_out)

