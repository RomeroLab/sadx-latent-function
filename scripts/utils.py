import os
import sys
import re

import Bio.SeqIO

PARENT_2D_DNA= 'ATGCAGCATACCTATCCGGCACAGCTGATGCGTTTTGGCACCGCAGCACGTGCAGAACATATGACCATTGCAGCAGCAATTCATGCACTGGATGCAGATGAAGCAGATGCAATTGTTATGGATATTGTTCCGGATGGTGAACGTGATGCATGGTGGGATGATGAAGGTTTTAGCAGCAGCCCGTTTACCAAAGATGCACATCATGCAGGAGTTGTTGCAACCAGCGTTACCCTGGGTCAGCTGCAGCGTGAACAGGGTGATAAACTGGTTAGCAAAGCAGCAGAATATTTTGGTATTGCCTGCCGTGTTAATGATGGTCTGCGTACCACCCGTTTTGTTCGTCTGTTTAGTGATGCCCTGGATGCCAAACCGCTGACCATTGGTCATGATTATGAAGTTGAATTTCTGCTGGCAACCCGTCGTGTTTATGAACCGTTTGAAGCACCGTTTAACTTTGCACCGCATTGTGGCGATGTTAGCTATGGTCGTGATACCGTTAATTGGCCTCTGAAACATAGCTTTCCGCGTCAGCTGGGTGGTTTTCTGACCATTCAGGGTGCAGATAATGATGCCGGTATGGTTATGTGGGATAATCGTCCGGAAAGCCGTGCAGCGCTGGATGAAATGCATGCAGAATATCGTGAAACCGGTGCAATTGCCGCACTGGAACGTGCAGCCAAAATCATGCTGAAACCGCAGCCTGGCCAGCTGACACTGTTTCAGAGCAAAAATCTGCATGCCATTGAACGTTGTACCAGCACCCGTCGTACCATGGGTCTGTTTCTGATTCATACCGAAGATGGTTGGCGTATGTTTGATTGA'

PARENT_2D_AA = "MQHTYPAQLMRFGTAARAEHMTIAAAIHALDADEADAIVMDIVPDGERDAWWDDEGFSSSPFTKDAHHAGVVATSVTLGQLQREQGDKLVSKAAEYFGIACRVNDGLRTTRFVRLFSDALDAKPLTIGHDYEVEFLLATRRVYEPFEAPFNFAPHCGDVSYGRDTVNWPLKHSFPRQLGGFLTIQGADNDAGMVMWDNRPESRAALDEMHAEYRETGAIAALERAAKIMLKPQPGQLTLFQSKNLHAIERCTSTRRTMGLFLIHTEDGWRMFD"

def parent2d_to_wt(p = PARENT_2D_AA):
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

WT_AA = parent2d_to_wt(PARENT_2D_AA)
WT = WT_AA

parent_map = {"2D":"2-D", "1VH":"1-VH", "2L":"2-L", "3VRL":"3-VRL"}

def get_parent_seq(parent, asdna=False):
    """ parent can be WT or 2D 
        or 1-VH, 2-L or 3-VRL
    """
    ret = None
    # get formal parent code from parent_map if it exists
    parent = parent_map.get(parent, parent)
    if parent == "WT":
        if asdna: raise NotImplementedError
        ret = WT_AA
    elif parent == "2-D":
        if asdna: raise NotImplementedError
        ret = PARENT_2D_AA
    elif parent in ("1-VH", "2-L",  "3-VRL"):
        ret = Bio.SeqIO.read(f"../data/{parent}.fasta", "fasta")
        if asdna:
            ret = str(ret.seq)
        else:
            ret = str(ret.translate().seq)
    return ret


def add_projects_to_path():
    # Modify the module path (if required) so that we can import
    # modules from this repository
    module_path = os.path.expanduser("~/sameerd/projects")
    if module_path not in sys.path:
        sys.path.append(module_path)

def add_openfold_to_path():
    # Modify the module path (if required) so that we can import
    # modules from this repository
    module_path = os.path.expanduser("~/software/openfold")
    if module_path not in sys.path:
        sys.path.append(module_path)

def add_alphafold_to_path():
    # Modify the module path (if required) so that we can import
    # modules from this repository
    module_path = os.path.expanduser("~/software/alphafold")
    if module_path not in sys.path:
        sys.path.append(module_path)




def hamming_dist(s1, s2):
    assert(len(s1) == len(s2))
    return sum(1 for (a, b) in zip(s1, s2) if a != b)

def mut_as_string(mut, wt, offset=0, mut_sep=";"):
    return mut_sep.join([f"{w}{i+offset}{m}" for
                i, (m,w) in enumerate(zip(mut, wt)) if m != w])


# ****************************************************************************
# convert shortened mutant sequence to full sequence
# this is not used anymore since we also save the full sequence
# but keeping this function incase we want it again
# It is the *inverse* of the function mut_as_string above
parse_mutant = re.compile(r"^(?P<wtaa>[A-Z])(?P<idx>[0-9]*)(?P<mutaa>[A-Z])$")

def expand_mut_str_list_to_list(mut_str, ref_list, encoder=lambda x: x, 
        split_mut_char=",", offset=1):
    """ Convert a mutation string like 'A13D;F131K' to a list of letters 
        or numbers
    >>> expand_mut_str_list_to_list("B2D,A3B", [0,1,0,1], 
    ...        encoder=lambda x: {'A':0, 'B':1, 'C':2, 'D':3}[x], offset=1)
    [0, 3, 1, 1]

    >>> expand_mut_str_list_to_list("", [0,1,0,1], offset=1)
    [0, 1, 0, 1]
    """
    mut_list = ref_list.copy()
    if isinstance(mut_str, str) and len(mut_str) > 0: 
        for m in mut_str.split(split_mut_char):
            md = parse_mutant.match(m).groupdict()
            idx = int(md['idx'])
            # check that we got the offset correct by maching the reference
            assert(encoder(md['wtaa']) == ref_list[idx - offset])
            mut_list[idx - offset] = encoder(md['mutaa'])
    else: # just return ref_list copy if no string passed in
        pass
    return mut_list


def expand_mut_str_list_to_seq(mut_str, ref, split_mut_char=";", offset=1):
    """ Convert a mutation string like 'A13D;F131K' to a full sequence
    >>> expand_mut_str_list_to_seq("B2D;A3B", "ABAB", offset=1)
    'ADBB'

    >>> expand_mut_str_list_to_seq("", "ABAB", offset=1)
    'ABAB'
    """
    ret = ref
    if mut_str != "*0*":
        mut = expand_mut_str_list_to_list(mut_str, list(ref), 
                encoder = lambda x: x,
                split_mut_char=split_mut_char, offset=offset)
        ret = "".join(mut)
    return ret
# ****************************************************************************

def get_columns_below_std_threshold(df, std_threshold=0.001):
    """ Looks at numeric columns and figures out which ones are below 
        the std threshold """
    df_std = df.std(numeric_only=True)
    return list(df_std[df_std < std_threshold].index)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
