import numpy as np
import energy

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
                        self.e_i_a_j_b)
        elif len(shape) == 2:
            if shape[1] != self.L:
                raise ValueError(f"MSA Protein needs {self.L} in 2nd dim."
                                 f" Shape : {shape}")
            else:
                ret = energy_calc_msa(protein, self.h_i_a, 
                        self.e_i_a_j_b)
        else:
            raise ValueError(f"input can only be dimension 1 or 2."
                             f" Shape : {shape}")
        return ret

 
def energy_calc_single(prot_np, h_i_a, e_i_a_j_b):
    """prot_np  : A numpy integer array of length L 
                  (with the correct Amino acid index in each position)
                  Maximum value prot_np should be 20
       h_i_a    : field values (shape = (L, q))
       e_i_a_j_b: coupling values (shape = (L, q, L, q))

       Returns : energy (float)
    """
    energy = energy_calc_msa(prot_np[np.newaxis, ...], h_i_a, e_i_a_j_b)
    return energy.squeeze()


def energy_calc_msa(msa, h_i_a, e_i_a_j_b):
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

    if msa.max() >= h_i_a.shape[1]: # bounds checking
        raise ValueError("Max value in msa should be less than alphabet size")
    return energy.energy_calc_msa(msa, h_i_a, e_i_a_j_b)


def energy_calc_single_mutants(seq, h_i_a, e_i_a_j_b):
    """seq      : A numpy integer array of shape (L,) 
                  (with the correct Amino acid index in each position)
                  Maximum value prot_np should be 20 (or q)
       h_i_a    : field values (shape = (L, q))
       e_i_a_j_b: coupling values (shape = (L, q, L, q))
       return a tuple (All single mutants, energies)
    """
    if seq.max() >= h_i_a.shape[1]: # bounds checking
        raise ValueError("Max value in seq should be less than alphabet size")
    return energy.energy_calc_single_mutants(seq.squeeze(), h_i_a, e_i_a_j_b)

def create_single_mutant(i, a, wt):
    mut = wt.copy()
    mut[i] = a
    return (mut)


if __name__ == "__main__":
    ## FIXME: Add unit tests
    pass


