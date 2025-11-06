import numpy as np
from . import _potts

empty_adj_mat = np.empty(shape=(0,0), dtype=int)

class EnergyFunctionCalculator:
    """ Calculate Energy for integer index arrays via __call__
        Can pass in either a single protein or an MSA 
    """

    def __init__(self, 
                 h_i_a, # local fields
                 e_i_a_j_b): # pairwise couplings
        self.h_i_a = h_i_a
        self.e_i_a_j_b = e_i_a_j_b
        self.L = self.h_i_a.shape[0]

    def __call__(self, protein, *args, **kwargs):
        """ We can pass in an array of numpy ints representing 
            a protein sequence or an MSA. 
            Array shape (L, ) gives an energy for that sequence
            Array shape (n_seq, L) returns an energy array of length
                n_seq for that MSA
        """
        shape = protein.shape
        ret = None
        if len(shape) == 1:
            if shape[0] != self.L:
                raise ValueError(f"Single Protein needs length {self.L}"
                                 f" Shape : {shape}")
            else:
                ret = energy_calc_single(protein, self.h_i_a, 
                        self.e_i_a_j_b, *args, **kwargs)
        elif len(shape) == 2:
            if shape[1] != self.L:
                raise ValueError(f"MSA Protein needs {self.L} in 2nd dim."
                                 f" Shape : {shape}")
            else:
                ret = energy_calc_msa(protein, self.h_i_a, 
                        self.e_i_a_j_b, *args, **kwargs)
        else:
            raise ValueError(f"input can only be dimension 1 or 2."
                             f" Shape : {shape}")
        return ret

 
def energy_calc_single(prot_np, h_i_a, e_i_a_j_b):
    """Calculate energy of protein using c++ 

       prot_np  : A numpy integer array of length L 
                  (with the correct Amino acid index in each position)
                  Maximum value prot_np should be 20
       h_i_a    : field values (shape = (L, q))
       e_i_a_j_b: coupling values (shape = (L, q, L, q))

       Returns : energy (float)
    """
    energy = energy_calc_msa(prot_np[np.newaxis, ...], h_i_a, e_i_a_j_b)
    return energy.squeeze()


def energy_calc_msa(msa, h_i_a, e_i_a_j_b, adj_mat=None):
    """msa     : A numpy integer array of shape (nseqs, L) 
                  (with the correct Amino acid index in each position)
                  Maximum value prot_np should be 20 (or q)
       h_i_a    : field values (shape = (L, q))
       e_i_a_j_b: coupling values (shape = (L, q, L, q))

       NOTE: this function is supposed to be a faster version of 
       energy_calc.energy_calc_msa. It does no real bounds checking so 
       we do some rudimentary checks before passing to C++
        
       Returns: energy (np float array of size nseqs)
    """
    if msa.size == 0: # incase an empty array is passed in
        # return an empty array
        # if we do not check this then msa.max() fails below
        return np.array([], dtype=float)
    if adj_mat is None:
        adj_mat = empty_adj_mat
    if msa.max() >= h_i_a.shape[1]: # bounds checking
        raise ValueError("Max value in msa should be less than alphabet size")
    return _potts.energy_calc_msa(msa, h_i_a, e_i_a_j_b, adj_mat)


def energy_calc_single_mutants(seq, h_i_a, e_i_a_j_b, adj_mat=None,
                                    recarray=True):
    """seq      : A numpy integer array of shape (L,) 
                  (with the correct Amino acid index in each position)
                  Maximum value prot_np should be 20 (or q)
       h_i_a    : field values (shape = (L, q))
       e_i_a_j_b: coupling values (shape = (L, q, L, q))
       return a tuple (All single mutants, energies)
    """
    if seq.max() >= h_i_a.shape[1]: # bounds checking
        raise ValueError("Max value in seq should be less than alphabet size")
    L, q = h_i_a.shape
    if adj_mat is None:
        adj_mat = empty_adj_mat
    muts, mut_energies = _potts.energy_calc_single_mutants(seq.squeeze(), 
                                    h_i_a, e_i_a_j_b, adj_mat)
    if recarray: # convert muts to recarray
        assert(muts.shape == (L * (q-1), 2) )
        muts = np.core.records.fromrecords(muts, names="i,a")
    return (muts, mut_energies)
    

def create_single_mutant(i, a, from_prot):
    mut = from_prot.copy()
    mut[i] = a
    return mut


def energy_calc_single_einsum(prot_np, h_i_a, e_i_a_j_b):
    """ Calculate energy of single mutant using einsum 
        An implementaton of energy_calc_single using np.einsum"""
    L, q= h_i_a.shape
    prot_one_hot = np.eye(q)[prot_np]
    e = e_i_a_j_b.copy() 

    # set the lower triangular part to zero (includes the diagonal)
    lower_i, lower_j = np.tril_indices(L)
    upper_i, upper_j = np.triu_indices(L)
    e[lower_i, :, lower_j, :] = 0

    # set the lower triangular part to zero
    prob = np.einsum("ia,ia", prot_one_hot,h_i_a, dtype=np.float64) +  \
            np.einsum("ia,iajb,jb",prot_one_hot,e,prot_one_hot, dtype=np.float64)
    return -prob

def energy_calc_single_python(prot_np, h_i_a, e_i_a_j_b):
    """ Calculate energy of single mutant using python loops 
        An implementaton of energy_calc_single using only python (slowest)"""
    L, q= h_i_a.shape

    main_h = main_e = 0
    for i in range(L):
        main_h += h_i_a[i, prot_np[i]]
    for i in range(L):
        for j in range(L):
            if i < j:
                main_e += e_i_a_j_b[i, prot_np[i], j, prot_np[j]]

    prob = main_h + main_e
    return -prob



if __name__ == "__main__":
    ## FIXME: Add doctests if unit tests don't cover everything
    pass


