""" Make mutations to protein arrays """
import numpy as np

from . import encoding as enc

def convert_wt_into_mutant_msa(wt, pos_idx, mut_values, seq_idx=None):
    """Convert WT into a msa of mutants

    params:
        wt          : Wildtype 1D np array dtype=np.uint8 
        pos_idx     : seq position to mutate
        mut_values  : mutate to these values dtype=np.uint8 
                      (same size as pos_idx)
        seq_idx     : index mutants using seq_idx instead of range(pos_idx)
                      This changes the number of mutants and allows for multiple
                      mutations in a sequence. If not provided, we assume that 
                      we are making an MSA of single mutants.
    returns:
        msa         : np array shape (max(seq_idx)+1, wt.size, dtype=np.uint8)
                      with mutations in the right place. If seq_idx is not
                      provided, then the first dimension of msa is pos_idx.size

    >>> wt = np.array([8, 17, 0, 5], dtype=np.uint8) 
    >>> pos_idx = np.array([2, 1, 1], dtype=int) # mutate these seqs
    >>> mut_values = np.array([3, 4, 7], dtype=np.uint8) #at these pos
    >>> convert_wt_into_mutant_msa(wt, pos_idx, mut_values) # single muts
    array([[ 8, 17,  3,  5],
           [ 8,  4,  0,  5],
           [ 8,  7,  0,  5]], dtype=uint8)
    >>> seq_idx = np.array([0, 2, 2], dtype=int) # multiple mutations
    >>> convert_wt_into_mutant_msa(wt, pos_idx, mut_values, seq_idx)
    array([[ 8, 17,  3,  5],
           [ 8, 17,  0,  5],
           [ 8,  7,  0,  5]], dtype=uint8)
    """
    # check shapes match up
    assert((pos_idx.ndim == 1)
            and (wt.ndim == 1)
            and (mut_values.ndim == 1)
            and (pos_idx.size == mut_values.size))

    if seq_idx is None: # make single mutations
        seq_idx = np.arange(pos_idx.size)
    
    msa = np.tile(wt, (seq_idx.size, 1)) # make copies of wt
    msa[seq_idx, pos_idx] = mut_values # assign multiple mutations if required
    return msa
    

def make_random_mutations(msa, seq_idx, pos_idx, 
        mutate_arr=enc.DEFAULT_ENCODER.int_mutate_arr, inplace=True):
    """Make random mutations in an msa at specific sequences and positions 
    params:
        msa     : np array of dtype uint8 shape (num_seqs, length)
        seq_idx: specifies which sequences to mutate. np array of any length
                  (repeats ok) but all values less than num_seqs 
        pos_idx: specifies which position to mutate at which index
    >>> msa = np.random.randint(21, size=(10, 100)) # 10 seqs, length 100
    >>> seq_idx = np.array([3, 3, 6], dtype=np.uint8) # mutate these seqs
    >>> pos_idx = np.array([20, 40, 60], dtype=np.uint8) #at these pos
    >>> mut_msa = make_random_mutations(msa, seq_idx, pos_idx,
    ...                         inplace=False)
    >>> (msa[seq_idx, pos_idx]  # check we got mutations
    ...         != mut_msa[seq_idx, pos_idx]).all()
    True
    >>> (msa[3, :] != mut_msa[3, :]).sum() # 2 mutations in seq number 3
    2
    >>> (msa[7:, :] == mut_msa[7:, :]).all() # check seq_idx >= 7 not mutated
    True
    """
    assert((seq_idx.ndim == 1)
            and (msa.ndim == 2)
            and (pos_idx.ndim == 1)
            and (seq_idx.size == pos_idx.size))
    mut_msa = msa
    if not inplace:
        mut_msa = msa.copy()
    # the values that need to be mutated
    mutate_idx = msa[seq_idx, pos_idx] 
    # pick random positions from the mutate table
    mutate_to = np.random.randint(mutate_arr.shape[1], size=mutate_idx.size)
    # set places we want to mutate to random other numbers from mutate
    # table. This ensures we actually get a mutant when where we want it
    mut_msa[seq_idx, pos_idx] = mutate_arr[mutate_idx, mutate_to]
    return mut_msa

if __name__ == "__main__":
    # to run these doc tests run them as a module 
    # > PYTHONPATH=".." ipython -m sequences.mutations
    import doctest
    doctest.testmod()


