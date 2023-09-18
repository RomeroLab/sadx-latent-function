import pandas as pd

import Bio
import Bio.SeqIO
import Bio.Seq


output_fa = "./inputs/seqs/SadA.fa"

if __name__ == "__main__":
    df = pd.read_csv("../rosetta/screening_data/"
                        "ordinal_Oct22_sequences_as_variant.csv")
    df["fasta_name"] = df.parent + "_" + \
                        df.variant.str.replace("\.", "_", regex=True)
    df["fasta_seq"] = "M" + df.sequence_aa_trim 


    with open(output_fa, "w") as fh:
        for fasta_seq, fasta_name in zip(df.fasta_seq, df.fasta_name):
            rec = Bio.SeqIO.SeqRecord(seq=Bio.Seq.Seq(fasta_seq), 
                    id=fasta_name, name=fasta_name)
            Bio.SeqIO.write(rec, fh, "fasta")
