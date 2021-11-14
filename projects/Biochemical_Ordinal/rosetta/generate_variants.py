#!/usr/bin/env python3

import random 

PROTEIN_LETTERS = "ACDEFGHIKLMNPQRSTVWY"
mutate_dict = {a:[b for b in PROTEIN_LETTERS if b != a] 
                        for a in PROTEIN_LETTERS}

PARENT_2D = "MQHTYPAQLMRFGTAARAEHMTIAAAIHALDADEADAIVMDIVPDGERDAWWDDEGFSSSPFTKDAHHAGVVATSVTLGQLQREQGDKLVSKAAEYFGIACRVNDGLRTTRFVRLFSDALDAKPLTIGHDYEVEFLLATRRVYEPFEAPFNFAPHCGDVSYGRDTVNWPLKHSFPRQLGGFLTIQGADNDAGMVMWDNRPESRAALDEMHAEYRETGAIAALERAAKIMLKPQPGQLTLFQSKNLHAIERCTSTRRTMGLFLIHTEDGWRMFD"

def parent2d_to_wt(p = PARENT_2D):
    # make reverse mutations to get to WT
    retlist = list(p)

    # D65N : 2D -> WT
    assert(p[65-1] == "D")
    retlist[65-1] = "N"
    # V71I : 2D -> WT
    assert(p[71-1] == "V")
    retlist[71-1] = "I"
    # H172R : 2D -> WT
    assert(p[172-1] == "H")
    retlist[172-1] = "R"
    # G157D : 2D -> WT
    assert(p[157-1] == "G")
    retlist[157-1] = "D"

    return "".join(retlist)

WT = parent2d_to_wt(PARENT_2D)


def make_random_point_mutation(s, pos):
    a = s[pos]
    b = random.choice(mutate_dict[a])
    # convert from python index to uniprot index by adding 1
    return a + str(pos+1) + b


if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--parent",
                    type=str,
                    choices=["WT", "PARENT_2D"], required=True,
                    help="PARENT to start with")
    parser.add_argument("-d", "--dist_parent",
                    type=int, required=True,
                    help="Distance from WT for each variant")
    parser.add_argument("-n", "--num_variants",
                    type=int, required=True,
                    help="Number of variants to generate")
    parser.add_argument("-s", "--seed",
                    type=int, default=111,
                    help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    parent = globals()[args.parent]
    n = args.num_variants
    d = args.dist_parent
    L = len(parent)

    # do not mutate the first position as it random
    positions_to_mutate = list(range(1, L))

    for _ in range(n):
        variant_positions = random.sample(positions_to_mutate, d)
        variant = ".".join([make_random_point_mutation(parent, pos)
                        for pos in variant_positions])
        print(variant)
        

