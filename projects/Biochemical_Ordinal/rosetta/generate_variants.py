import warnings

import Bio.SeqIO
import Bio.Data.IUPACData
from Bio import BiopythonParserWarning

protein_letters = Bio.Data.IUPACData.protein_letters

def wt_seq_from_pdb(filename):
    ret = None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', BiopythonParserWarning)
        ret = Bio.SeqIO.read(filename, "pdb-atom")
        ret = str(ret.seq)
    return ret

def generate_single_mutants(wt_seq):
    for i, wt_aa in enumerate(wt_seq):
        for mut_aa in protein_letters:
            if mut_aa == wt_aa:
                continue
            else:
                yield f"{wt_aa}{i+1}{mut_aa}"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pdb",
                    type=argparse.FileType('r'),
                    required=True,
                    help="PARENT to start with")
    args = parser.parse_args()

    wt_seq = wt_seq_from_pdb(args.pdb)
    # let's include the WT sequence as the first variant
    # This is useful to compare other energies to
    print(f"X1{wt_seq[0]}")
    for variant in generate_single_mutants(wt_seq):
        print(variant)
