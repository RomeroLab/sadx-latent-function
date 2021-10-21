import os
import sys
import re

def add_projects_to_path():
    # Modify the module path (if required) so that we can import
    # modules from this repository
    module_path = os.path.expanduser("~/sameerd/projects")
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

def expand_mut_str_list(mut_str, ref, split_mut_char=";", offset=1):
    """ Convert a mutation string like 'A13D;F131K' to a full sequence"""
    mut = list(str(ref))
    if isinstance(mut_str, str) and len(mut_str) > 0: 
        for m in mut_str.split(split_mut_char):
            md = parse_mutant.match(m).groupdict()
            idx = int(md['idx'])
            # check that we got the offset correct by maching the reference
            assert(md['wtaa'] == ref[idx - offset])
            mut[idx - offset] = md['mutaa']
    else: # we are going to return the reference as we don't have any mutants
        pass
    return "".join(mut)
# ****************************************************************************
