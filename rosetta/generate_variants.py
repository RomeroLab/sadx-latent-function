import warnings

import Bio.SeqIO
import Bio.Data.IUPACData
from Bio import BiopythonParserWarning

protein_letters = Bio.Data.IUPACData.protein_letters

def wt_seq_from_pdb(filename, chain="A"):
    ret = None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', BiopythonParserWarning)
        for record in Bio.SeqIO.parse(filename, "pdb-atom"):
            if record.annotations["chain"] == chain:
                ret = str(record.seq)
                break
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
    parser.add_argument("-c", "--chain",
                    type=str,
                    default="A",
                    help="Which chain to use")
    args = parser.parse_args()

    wt_seq = wt_seq_from_pdb(args.pdb, chain=args.chain)
    # let's include the WT sequence as the first variant
    # This is useful to compare other energies to
    print(f"X1{wt_seq[0]}")
    for variant in generate_single_mutants(wt_seq):
        print(variant)
