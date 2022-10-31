import pathlib
import numpy as np
import pandas as pd

import Bio
import Bio.Seq
import Bio.SeqIO

import pysam


import matplotlib.pyplot as plt
import seaborn as sns


run_number = "r64120_20221013_213137"
demux_run = "m64120_221016_144219"
data_dir = pathlib.Path(f"../data/scratch/{run_number}/3_C01")


WT_fasta = Bio.SeqIO.read("../data/sad1VH.fasta", format="fasta")
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


def shorten_seq(s, wt=WT_DNA):
    assert(len(s) == len(wt)) # don't do deletions for now
    ret = ",".join(f"{wi}{i+1}{si}" for i, (si, wi) 
                   in enumerate(zip(s, wt)) if wi != si)
    if not len(ret): # then we have wild-type
        ret = f"*0*" # special return for wild-type
    return ret
    

if __name__ == "__main__":
    output_dir = pathlib.Path("../data/scratch/Oct22")
    samples = pd.read_csv(data_dir / "00Samples", sep="\t", skiprows=1)
    samples_counts = output_dir / "Samples.csv"
    with open(samples_counts, "wt") as fh_out:
        print("Barcode,Bio_sample,num_sequences,num_kept", file=fh_out)
 
    for b_idx, barcode in enumerate(samples.Barcode):
        bamfile = data_dir / "demux" / f"{demux_run}.demux.{barcode}--{barcode}.bam"
        print(f"Processing : {bamfile}")
        keep_seqs = []
        samfile = pysam.AlignmentFile(bamfile, "rb", check_sq=False)
        for i, line in enumerate(samfile):
            if len(line.seq) >= end: # length filter
                qual = np.array(line.get_forward_qualities()[start:end])
                if not np.any(qual < min_phred_base): # quality filter
                    # check to see if we should reverse transcript
                    rc_better = False
                    seq = line.seq[start:end]
                    ref_dist = hamming_dist(seq, WT_DNA)
                    if ref_dist > 100: # also check the reverse complement
                        r_seq = Bio.Seq.Seq(line.seq).reverse_complement()[start:end]
                        r_ref_dist = hamming_dist(r_seq, WT_DNA)
                        if r_ref_dist < ref_dist:
                            rc_better = True
                            seq = r_seq
                            ref_dist = r_ref_dist
                    ss = shorten_seq(Bio.Seq.Seq(seq).translate(), WT)
                    keep_seqs.append([ss, ref_dist, rc_better])
            #if i > 1000:
            #    break
        samfile.close()
        keep_seqs = pd.DataFrame(keep_seqs, columns=["ss", "aa_dist", "rc_better"])     
        keep_seqs.to_csv(output_dir / f"{barcode}.tsv", sep="\t", index=False)
        with open(samples_counts, "a") as fh_out:
            print(f"{barcode},{samples.Bio_Sample[b_idx]},{i},{len(keep_seqs)}", file=fh_out)




