def hamming_dist(s1, s2):
    assert(len(s1) == len(s2))
    return sum(1 for (a, b) in zip(s1, s2) if a != b)

def mut_as_string(mut, wt, offset=0, mut_sep=","):
    return mut_sep.join([f"{w}{i+offset}{m}" for
                i, (m,w) in enumerate(zip(mut, wt)) if m != w])
