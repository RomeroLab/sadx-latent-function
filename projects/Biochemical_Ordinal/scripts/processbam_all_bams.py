import pathlib
import numpy as np
import pandas as pd

import Bio
import Bio.Seq
import Bio.SeqIO
from Bio import Align

import pysam


import matplotlib.pyplot as plt
import seaborn as sns


run_number = "r64120_20221013_213137"
demux_run = "m64120_221016_144219"
data_dir = pathlib.Path(f"../data/scratch/{run_number}/3_C01")


# This is SadX without the MBP part (the OG wild-type)
SAD_WT_DNA = Bio.SeqIO.read("../data/SadA D157G.fasta", format="fasta").seq
SAD_WT_AA = SAD_WT_DNA.translate()

PARENT_1VH_DNA = Bio.SeqIO.read("../data/1-VH.fasta", format="fasta").seq
PARENT_2L_DNA = Bio.SeqIO.read("../data/2-L.fasta", format="fasta").seq
PARENT_3VRL_DNA = Bio.SeqIO.read("../data/3-VRL.fasta", format="fasta").seq

parent_d = {
        "1VH":PARENT_1VH_DNA,
        "2L":PARENT_2L_DNA,
        "3VRL":PARENT_3VRL_DNA
        }

start = 53
end = 875

length_cutoff = 1
min_phred_base = 93
# https://ccs.how/faq/reads-bam.html#how-to-get-hifi-reads
#min_read_quality = 0.99 # minimum read quality
mut_cut = 15


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
    samples = pd.read_csv(data_dir / "00Samples", sep="\t", skiprows=1)

    # write out sample counts to this file
    samples_counts = output_dir / "Samples.csv"
    with open(samples_counts, "wt") as fh_out:
        print("Barcode,Bio_sample,num_sequences,num_kept,"
              "passed_length_filter,passed_quality_filter,"
              "samples_placed", file=fh_out)
 

    aligner = Align.PairwiseAligner()
    aligner.query_left_open_gap_score = -10. 
    aligner.query_left_extend_gap_score = -0.5
    aligner.query_right_open_gap_score = -10.
    aligner.query_right_extend_gap_score = -0.5
    aligner.target_internal_gap_score = -50.
    aligner.query_internal_gap_score = -50.
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.mode = 'global'


    for b_idx, barcode in enumerate(samples.Barcode):
        bamfile = data_dir / "demux" / f"{demux_run}.demux.{barcode}--{barcode}.bam"

        bio_sample = samples.Bio_Sample[b_idx]
        parent = bio_sample[:bio_sample.find("_")]
        print(f"Processing : {bamfile}, {bio_sample} {parent}")

        wt_dna = parent_d[parent]
        wt = wt_dna.translate()

        keep_seqs = []
        save = pysam.set_verbosity(0)
        samfile = pysam.AlignmentFile(bamfile, "rb", check_sq=False)
        pysam.set_verbosity(save)

        length_filter_counter = 0
        quality_filter_counter = 0
        for i, line in enumerate(samfile):
            if len(line.seq) >= end: # length filter
                length_filter_counter += 1
                qual = np.array(line.get_forward_qualities()[start:end])
                if not np.any(qual < min_phred_base): # quality filter
                #if line.get_tag('rq') > min_read_quality: # quality filter
                    quality_filter_counter += 1
                    # check to see if we should reverse transcript
                    rc_better = False 
                    seq = line.seq[start:end] 
                    ref_dist = hamming_dist(seq, wt_dna)
                    if ref_dist > 100: # also check the reverse complement
                        r_line = Bio.Seq.Seq(line.seq).reverse_complement()
                        r_seq = r_line[start:end]
                        r_ref_dist = hamming_dist(r_seq, wt_dna)
                        if r_ref_dist < ref_dist:
                            rc_better = True
                            seq = r_seq
                            ref_dist = r_ref_dist
                        if ref_dist > 100: # still greater than 100
                            with open(f"{output_dir}/{barcode}.long.csv", "a") as fh_long:
                                print(f"{line.seq},{wt_dna}", file=fh_long)
                            # see if we can align better to wt_dna
                            rc_better = False # lets check this again after alignment
                            o_line = line.seq # original line sequence
                            a = aligner.align(wt_dna, o_line)
                            r_a = aligner.align(wt_dna, r_line)
                            if (a.score < r_a.score): # reverse complement alignes better
                                a = r_a
                                o_line = r_line
                                rc_better = True
                            b = a[0] # first alignement
                            # look at the aligned chunks and see where the first one is
                            aligned_start = b.aligned[1][0][0] 
                            aligned_end = aligned_start + (end - start)
                            if (len(o_line) - aligned_start) >= len(wt_dna):
                                seq = o_line[aligned_start:aligned_end]
                            #break
                            #pass
                    ss_dna = shorten_seq(seq, wt_dna)
                    dna_dist = hamming_dist(seq, wt_dna)

                    seq_translate = Bio.Seq.Seq(seq).translate()
                    ss_aa = shorten_seq(seq_translate, wt)
                    aa_dist = hamming_dist(seq_translate, wt) 

                    keep_seqs.append([ss_dna, dna_dist, ss_aa, aa_dist, 
                            ref_dist, rc_better])
            if TESTING and (i > 1000):
                break
        #break
        samfile.close()
        keep_seqs = pd.DataFrame(keep_seqs, 
                    columns=["ss_dna", "dna_dist", "ss_aa", "aa_dist",
                             "ref_dist", "rc_better"])     
        keep_seqs.to_csv(output_dir / f"{barcode}.tsv", sep="\t", index=False)
        num_samples_placed = num_samples_placed_d[bio_sample]
        with open(samples_counts, "a") as fh_out:
            print(f"{barcode},{bio_sample},{i},{len(keep_seqs)},"
                  f"{length_filter_counter},{quality_filter_counter},"
                  f"{num_samples_placed}", file=fh_out)


