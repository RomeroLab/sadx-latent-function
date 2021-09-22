import timeit
import numpy as np
import protein_utils.energy.potts as energy_py

rng = np.random.default_rng(100)

def calc_energy_msa_using_single(msa, h, e):
    return np.apply_along_axis(
            lambda x: energy_py.energy_calc_single(x, h, e), 
            axis=1, arr=msa)


if __name__ == "__main__":
    L = 186
    q = 20
    num_seqs = 100
    h = rng.random(size=(L,q))
    e = rng.random(size=(L,q,L,q))
    msa = rng.integers(q, size=(num_seqs, L))
    wt = msa[0, :].squeeze()

    i_idx, j_idx = np.triu_indices(L,k=1)

    # pick out 2k interactions
    int_select  = rng.choice(i_idx.size, 2000, replace=False)
    int_select.sort()

    int_mask = np.zeros(i_idx.size, dtype=bool)
    int_mask[int_select] = True
    e_sparse = e.copy()
    e_sparse[i_idx[int_mask], :, j_idx[int_mask], :] = 0

    adj_mat = np.zeros(shape=(L,L), dtype=int)
    adj_mat[i_idx[int_mask], j_idx[int_mask]] = 1
    adj_mat[j_idx[int_mask], i_idx[int_mask]] = 1

    print("Timings are normalized so that they are per energy calculation")
    print()
    ret = timeit.timeit('energy_py.energy_calc_msa(msa, h, e)', 
            globals=globals(), number=10)
    print(f"Calc MSA c++ (direct)       : {ret/10/num_seqs*1000:7.2f}ms")
    ret = timeit.timeit('energy_py.energy_calc_msa(msa, h, e_sparse, adj_mat)', 
            globals=globals(), number=10)
    print(f"Calc MSA c++ (interaction)  : {ret/10/num_seqs*1000:7.2f}ms")
    ret = timeit.timeit('calc_energy_msa_using_single(msa, h, e)', 
            globals=globals(), number=10)
    print(f"Calc MSA c++ (using single) : {ret/10/num_seqs*1000:7.2f}ms")
    ret = timeit.timeit('calc_energy_msa_using_single(msa, h, e)', 
            globals=globals(), number=10)
    print(f"Calc MSA c++ (using single) : {ret/10/num_seqs*1000:7.2f}ms")
    print()
    ret = timeit.timeit('energy_py.energy_calc_single(wt, h, e)', 
            globals=globals(), number=1000)
    print(f"Calc WT c++     : {ret/num_seqs*1000:7.2f}ms")
    ret = timeit.timeit('energy_py.energy_calc_single_einsum(wt, h, e)', 
            globals=globals(), number=10)
    print(f"Calc WT einsum  : {ret/10*1000:7.2f}ms")
    ret = timeit.timeit('energy_py.energy_calc_single_python(wt, h, e)', 
            globals=globals(), number=10)
    print(f"Calc WT python  : {ret/10*1000:7.2f}ms")

    print("\nCalculating single mutants")
    ret = timeit.timeit('energy_py.energy_calc_single_mutants(wt, h, e)', 
            globals=globals(), number=10)
    print(f"Calc Single mutants (direct)       : {ret/10*1000:7.2f}ms")
    ret = timeit.timeit('energy_py.energy_calc_single_mutants(wt, h, e_sparse, adj_mat)', 
            globals=globals(), number=10)
    print(f"Calc Single mutants (interactions) : {ret/10*1000:7.2f}ms")

