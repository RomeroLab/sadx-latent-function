import numpy as np

from ..utils.fileio import file_handle_opener

def read_plmc_coupling_scores(filename, sort=True):
    """Reads in the coupling scores file of the plmc program and returns an
       array of scores in zero-based indexing. 

       filename : can be a filehandle, a regular or gzipped file. For gzip
                    make sure the extension ends in ".gz"
       sort     : sort results in descending order
    """
    coupling_scores = []
    opener = file_handle_opener(filename)
    
    with opener(filename, "rt") as fh: # open in text
        for line in fh:
            res_i, focus_i, res_j, focus_j, zero, score = line.strip().split()
            coupling_scores.append((int(res_i)-1, int(res_j)-1, float(score)))
    scores = np.array(coupling_scores, dtype=
                [('idx1', int), ('idx2', int), ('score', float)])
    if sort: # sort inplace and in descending order
        # numpy doesn't allow sorts in descending order. So we reverse the order
        # of the recarray and then sort in place. How exactly does this work???
        scores[::-1].sort(order="score")
    return(scores)
