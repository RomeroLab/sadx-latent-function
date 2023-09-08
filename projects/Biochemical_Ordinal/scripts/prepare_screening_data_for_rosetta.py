import pandas as pd
import utils


output_dir = "../rosetta/screening_data"

if __name__ == "__main__":
    df = pd.read_csv("../output/ordinal_Oct22_sequences_with_dca_score.csv")
    print(f"Read in {len(df)} sequences")
    df["sequence_aa"] = "M" + df.sequence_aa_trim + "*"


    parent_seqd = {parent:utils.get_parent_seq(parent) 
            for parent in df.parent.unique()}

    # make a variant column in the format that the
    # ../rosetta/create_variant_xml.sh script wants
    df["variant"] = df[["parent", "sequence_aa"]].apply(lambda x: 
            utils.mut_as_string(mut=x["sequence_aa"], 
                                wt=parent_seqd[x["parent"]], 
                                offset = 1,
                                mut_sep="."), 
            axis=1)

    # there should be three parent sequences
    assert(sum(df.variant == "") == len(parent_seqd))
    # set the "variant" column for the parent seqs to X1M 
    # this will ensure that no change is made to the parent sequence
    df.loc[df.variant == "", "variant"] = "X1M"

    df[["parent", "variant", "sequence_aa_trim"]].to_csv(
            f"{output_dir}/ordinal_Oct22_sequences_as_variant.csv",
            index=False)

    for parent in parent_seqd:
        parent_variants = df.loc[df.parent == parent, "variant"]
        parent_variants.to_csv(
            f"{output_dir}/ordinal_Oct22_sequences_variants_for_{parent}.txt",
            index=False, header=False)


