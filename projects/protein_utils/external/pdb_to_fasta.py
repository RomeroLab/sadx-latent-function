import Bio.SeqIO


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_pdb_filename",
                    help="input pdb filename",
                    required=True)
    args = parser.parse_args()

    with open(args.input_pdb_filename) as fh:
        for record in Bio.SeqIO.parse(fh, "pdb-atom"):
            print(">" + record.id)
            print(record.seq)






