import unittest

from pathlib import Path
THIS_DIR = Path(__file__).parent

import numpy as np
import protein_utils.potts_energy.energy_py as energy_py

np.random.seed(100)

class TestEnergyCalc(unittest.TestCase):

    def setUp(self):
        self.N = 10 # number of sequences
        self.L = 15 # length of protein
        self.q = 20 # size of alphabet

        self.msa = np.random.randint(low=0, high=self.q, size=(self.N,self.L))
        self.wt = self.msa[0, :].squeeze()

        # assign main effects
        self.h_i_a = np.random.random(size=(self.L, self.q))
        self.h_i_a = np.zeros_like(self.h_i_a)

        # assign interaction effects 
        # Only the upper triagular part of e_i_a_j_b will be used
        self.e_i_a_j_b = np.random.random(size=(self.L, self.q, self.L, self.q))


        ### set the lower triangular part to zero (includes the diagonal)
        #lower_i, lower_j = np.tril_indices(self.L)
        #upper_i, upper_j = np.triu_indices(self.L)
        #self.e_i_a_j_b[lower_i, :, lower_j, :] = 0


    def test_energy_calc_single(self):
        """ Check that C++, np and python versions of calculating energy all match up"""
        e_cpp = energy_py.energy_calc_single(self.wt, self.h_i_a, self.e_i_a_j_b)
        e_einsum = energy_py.energy_calc_single_einsum(self.wt, self.h_i_a, 
                                                self.e_i_a_j_b)
        e_python = energy_py.energy_calc_single_python(self.wt, self.h_i_a, 
                                                self.e_i_a_j_b)
        self.assertAlmostEqual(e_cpp, e_einsum)
        self.assertAlmostEqual(e_cpp, e_python)

    def test_single_mutants(self):
        muts, energy = energy_py.energy_calc_single_mutants(self.wt, 
                                    self.h_i_a, self.e_i_a_j_b)

        # make the first mutant manually    
        mut = self.wt.copy()
        mut[muts[0][0]] = muts[0][1] 
        # check energy of first mutant
        e_mut = energy_py.energy_calc_single(mut, self.h_i_a, self.e_i_a_j_b)
        self.assertAlmostEqual(e_mut, energy[0])



if __name__ == '__main__':
    unittest.main()





    







