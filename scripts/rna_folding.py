"""
See if RNA folding score correlates with ordinal regression

Conclusion: No correlation to very weak correlation

"""
import sys
import pandas as pd

import scipy.stats

add_path = "/usr/local/lib/python3.9/site-packages"
if add_path not in sys.path:
    sys.path.append(add_path)
import RNA

import utils

if __name__ == "__main__":
    seqs = pd.read_csv("../data/sequences_Oct22.tsv", sep="\t")
    seqs_no_dups = seqs.drop_duplicates(
                        ["parent", "category", "ss_dna"], keep='first')
    parent_seqs = (seqs_no_dups[seqs_no_dups.ss_dna == "*0*"]
                    .groupby("parent").head(1) )
    parent_seqs.category = "P"
    final_seqs = pd.concat(
            [seqs_no_dups[seqs_no_dups.ss_dna != "*0*" ], parent_seqs])
    final_seqs["response"] = final_seqs.category.map(
                                    {"H":3, "P":2, "L":1, "N":0})

    parent_seqs = {parent: utils.get_parent_seq(parent) 
                        for parent in final_seqs.parent.unique()}

    # map the shortened sequences to full sequences
    final_seqs["sequence_dna"] = final_seqs[["ss_dna", "parent"]].apply(
            lambda x: utils.expand_mut_str_list_to_seq(
                         x['ss_dna'], parent_seqs[x['parent']], 
                         split_mut_char=",", offset=1), axis=1) 
            
    final_seqs["rna_fold_score"] = final_seqs.sequence_dna.map(
            lambda x: RNA.fold(x)[1])

    # correlate rna folding score with response
    print(final_seqs.groupby(["parent"]).apply(
            lambda x: scipy.stats.spearmanr(-x.response, 
                        x.rna_fold_score).statistic))
    #parent
    #1VH     0.007654
    #2L      0.096740
    #3VRL    0.042366

    def f(x):
        d = {}

        sr = scipy.stats.spearmanr(-x.response, x.rna_fold_score)
        d['rna_spearman'] = sr.statistic
        d['rna_spearman_p'] = sr.pvalue

        d['N'] = len(x)
        return pd.Series(d, index=['rna_spearman', 'rna_spearman_p', 
                               'N'])

    parent_dna_dist_stats = final_seqs.groupby(["parent", "dna_dist"]).apply(f)
    parent_dna_dist_stats[parent_dna_dist_stats.N > 50]
    #                 rna_spearman  rna_spearman_p      N
    #parent dna_dist                                     
    #1VH    2            -0.081053        0.483452   77.0
    #       3            -0.090262        0.389544   93.0
    #       4            -0.050876        0.635867   89.0
    #       5            -0.022166        0.860872   65.0
    #2L     1             0.188144        0.181643   52.0
    #       2            -0.000739        0.994519   89.0
    #       3            -0.113523        0.215047  121.0
    #       4             0.072266        0.491211   93.0
    #       5            -0.050157        0.677856   71.0
    #       6             0.147932        0.300217   51.0
    #3VRL   3            -0.056724        0.579048   98.0
    #       4             0.112142        0.230719  116.0
    #       5            -0.003936        0.966405  117.0
    #       6             0.053412        0.649023   75.0


