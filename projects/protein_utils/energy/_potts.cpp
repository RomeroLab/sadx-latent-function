#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// dtype of the msa or seqs passed in
typedef long seq_dtype; // change this later to uint8?
typedef int adj_dtype; // change this later to uint8?

template <typename T1, typename T2>
void assert_equal(T1 x, T2 y, std::string msg) {
  if (x != y) 
    throw std::runtime_error(msg);      
}

void check_canonical_parameter_shapes(py::array_t<double> h_i_a,
                                      py::array_t<double> e_i_a_j_b) {
    // Doesn't seem like we need to pass by reference to this function as
    // somehow it isn't making a copy of the arrays
    auto buf_h = h_i_a.unchecked<2>(); 
    auto buf_e = e_i_a_j_b.unchecked<4>();

    assert_equal(buf_h.ndim(), 2, "Number of dims for fields must be 2");
    assert_equal(buf_e.ndim(), 4, "Number of dims for couplings must be 4");
    assert_equal(buf_h.shape(0), buf_e.shape(0), "couplings and fields "
                                  "do not match along the length dimension");
    assert_equal(buf_e.shape(0), buf_e.shape(2), "couplings do not match "
                                  "along the length dimension");
    assert_equal(buf_h.shape(1), buf_e.shape(1), "couplings and fields "
                                  "do not match along the alphabet dimension");
    assert_equal(buf_e.shape(1), buf_e.shape(3), "couplings do not match "
                                  "along the alphabet dimension");
}


template <typename T>
class SeqFromMSA {
  // Takes a sequence idx and a pointer to the MSA and allows
  // the operator () to directly access the sequence with one index only
  private:
    T m_ptr_;
    int m_seq_idx;
    
  public:
    size_t operator()(size_t idx) {return m_ptr_(m_seq_idx, idx);}
    SeqFromMSA(T ptr, int seq_idx) : m_ptr_(ptr), m_seq_idx(seq_idx) {}

};

// Potts energy calculator 
template <typename T>
double potts_energy(T& w,  // any type that implements () to give us the index
                py::detail::unchecked_reference<double,2>& buf_h,
                py::detail::unchecked_reference<double,4>& buf_e) {
  size_t L = (unsigned) buf_h.shape(0);
  double energy(0.);
  for (size_t i = 0; i < L ; i++) {
    // loop over each residue position
    auto i_idx = w(i); // AA index
    energy += buf_h(i, i_idx );
    for (size_t j = i+1; j < L; j++) {
      // double loop over residue position
      auto j_idx = w(j); // AA index
      energy += buf_e(i, i_idx, j, j_idx);
    }
  }
  return -energy;
}

template <typename T>
double potts_energy(T& w,  // any type that implements () to give us the index
                py::detail::unchecked_reference<double,2>& buf_h,
                py::detail::unchecked_reference<double,4>& buf_e,
                std::vector<std::vector<unsigned int>>& interactions) {
  size_t L = (unsigned) buf_h.shape(0);
  double energy(0.);
  for (size_t i = 0; i < L ; i++) {
    // loop over each residue position
    auto i_idx = w(i); // AA index
    energy += buf_h(i, i_idx );
    for (auto j : interactions[i]) {
      // double loop over residue position
      auto j_idx = w(j); // AA index
      energy += buf_e(i, i_idx, j, j_idx);
    }
  }
  return -energy;
}

double potts_energy_i_only(size_t i, seq_dtype a,  
                py::detail::unchecked_reference<seq_dtype,1>& buf_w,
                py::detail::unchecked_reference<double,2>& buf_h,
                py::detail::unchecked_reference<double,4>& buf_e) {
  size_t L = (unsigned) buf_h.shape(0);
  double energy_i = buf_h(i, a);
  for (size_t j = 0; j < L; j++) {
      if (j == i) continue;
      if (j < i) {
          energy_i += buf_e(j, buf_w(j), i, a);
      } else {
          energy_i += buf_e(i, a, j, buf_w(j));
      }
  }
  return -energy_i;
}

double potts_energy_i_only(size_t i, seq_dtype a,  
                py::detail::unchecked_reference<seq_dtype,1>& buf_w,
                py::detail::unchecked_reference<double,2>& buf_h,
                py::detail::unchecked_reference<double,4>& buf_e, 
                std::vector<std::vector<unsigned int>>& interactions) {
  double energy_i = buf_h(i, a);
  for (auto j : interactions[i]) {
      if (j < i) {
          energy_i += buf_e(j, buf_w(j), i, a);
      } else {
          energy_i += buf_e(i, a, j, buf_w(j));
      }
  }
  return -energy_i;
}


// convert adjacency matrix into std::vector of vectors
bool convert_to_interactions(py::array_t<adj_dtype> adj_mat, size_t L, 
                          std::vector<std::vector<unsigned int>>& interactions,
                          bool upper_tri) /* only upper tri interactions */ {
  bool valid_interactions(false);
  auto buf_a = adj_mat.unchecked<2>();
  if (buf_a.shape(0)) {
    assert_equal(buf_a.shape(0), buf_a.shape(1), "adj_mat must be a square "
                                                 "matrix");
    assert_equal(buf_a.shape(0), L, "msa and fields do not match "
                                    "along the length dimension");
    interactions.resize(L);
    // add indices j that interact with index i in interactions[i]
    if (upper_tri) {
      for(size_t i=0; i < L; ++i) {
        interactions[i].reserve(L - (i+1));
        for (size_t j=i+1; j < L; ++j) {
          if (buf_a(i,j)) interactions[i].push_back(j);
        }
      } // converted upper tri matrix into vectors
    } else { // do full adj matrix instead of upper tri
      for(size_t i=0; i < L; ++i) {
        interactions[i].reserve(L - 1);
        for (size_t j=0; j < L; ++j) {
          if (j == i) continue;
          if (buf_a(i,j)) interactions[i].push_back(j);
        }
      }
    } // finished doing upper and lower triangular matrix
    valid_interactions = true;
  }
  return valid_interactions;
}

py::array_t<double> energy_calc_msa(
           py::array_t<seq_dtype> msa, /* shape (num_seqs, L) */
           py::array_t<double> h_i_a,  /* shape (L, q) */
           py::array_t<double> e_i_a_j_b, /* shape (L, q, L, q) */ 
           py::array_t<adj_dtype> adj_mat) /* shape (L, L) */ {

    check_canonical_parameter_shapes(h_i_a, e_i_a_j_b);

    auto buf_m = msa.unchecked<2>(); 
    auto buf_h = h_i_a.unchecked<2>(); 
    auto buf_e = e_i_a_j_b.unchecked<4>();

    size_t L = (unsigned) buf_h.shape(0);
    assert_equal(buf_m.shape(1), L, "msa and fields do not match "
                                    "along the length dimension");

    std::vector<std::vector<unsigned int>> interactions;
    bool valid_interactions = convert_to_interactions(adj_mat, L, 
                                                        interactions, true);
    size_t num_seqs = (unsigned) buf_m.shape(0);

    // Resulting Energy Array
    auto result = py::array_t<double>(num_seqs);
    auto buf_r = result.request();
    double *ptr_r = (double *) buf_r.ptr;
    
    // loop over all sequences one at a time
    for (size_t seq_idx = 0; seq_idx < num_seqs; seq_idx++) {
        typedef SeqFromMSA<py::detail::unchecked_reference<seq_dtype, 2>> 
                    seq_from_msa;
        auto w = seq_from_msa(buf_m, seq_idx);
        double energy;  
        if (valid_interactions) {
          energy = potts_energy<seq_from_msa>(w, buf_h, buf_e, interactions);
        } else {
          energy = potts_energy<seq_from_msa>(w, buf_h, buf_e);
        }
        ptr_r[seq_idx] = energy;
    }

    return result;
}


py::tuple energy_calc_single_mutants(
                py::array_t<seq_dtype> wt, /* shape (L, ) */
                py::array_t<double> h_i_a,  /* shape (L, q) */
                py::array_t<double> e_i_a_j_b, /* shape (L,q,L,q) */
                py::array_t<adj_dtype> adj_mat) /* shape (L,L) */  {

    check_canonical_parameter_shapes(h_i_a, e_i_a_j_b);

    auto buf_w = wt.unchecked<1>(); 
    auto buf_h = h_i_a.unchecked<2>(); 
    auto buf_e = e_i_a_j_b.unchecked<4>();

    if (buf_w.shape(0) != buf_h.shape(0))
        throw std::runtime_error("sequence and fields do not match "
                                 "along the length dimension");

    size_t L = (unsigned) buf_h.shape(0);
    size_t q = (unsigned) buf_h.shape(1);

    std::vector<std::vector<unsigned int>> interactions;
    bool valid_interactions = convert_to_interactions(adj_mat, L, 
                                                        interactions, false);

    // Resulting Energy Array
    size_t num_mutants = L * (q-1);
    auto mut_shape = pybind11::array::ShapeContainer({(long) num_mutants, 2});
    auto mutants = py::array_t<long>(mut_shape);
    auto buf_m = mutants.request();
    auto writer_m = mutants.mutable_unchecked<2>();

    auto energies = py::array_t<double>(num_mutants);
    auto buf_en = energies.request();
    auto writer_en = energies.mutable_unchecked<1>();

    // Calculate the energy of the sequence passed in
    typedef py::detail::unchecked_reference<seq_dtype, 1> plain_seq;
    double energy_wt = potts_energy<plain_seq>(buf_w, buf_h, buf_e);
   
    // Now calculate the energy of the single mutants
    size_t mutant_counter = 0;
    for (size_t i = 0 ; i < L; i++) {
        // loop over each residue position
        auto wt_i = (unsigned) buf_w(i); // AA index of sequence
        double energy_wt_i;
        if (valid_interactions) {
          energy_wt_i = potts_energy_i_only(i, wt_i, buf_w, buf_h, buf_e,
                            interactions);
        } else {
          energy_wt_i = potts_energy_i_only(i, wt_i, buf_w, buf_h, buf_e);
        }

        for (size_t a = 0; a < q; a++) {
            if (a == wt_i) continue; // this is wt and not a mutant
            double energy_mut_i;
            if (valid_interactions) {
              energy_mut_i = potts_energy_i_only(i, a, buf_w, buf_h, buf_e,
                                interactions);
            } else {
              energy_mut_i = potts_energy_i_only(i, a, buf_w, buf_h, buf_e);
            }

            if (mutant_counter >= num_mutants) {
                throw std::runtime_error("mutant counter larger than"
                                " num_mutants"); 
            }
            writer_m(mutant_counter, 0) = i;
            writer_m(mutant_counter, 1) = a;
            auto energy_mut = energy_mut_i - energy_wt_i + energy_wt; 
            writer_en(mutant_counter) = energy_mut;

            ++mutant_counter;
        }
    }

    return py::make_tuple(mutants, energies);
}

PYBIND11_MODULE(_potts, m) {
    m.doc() = "pybind11 Energy calculator"; // optional module docstring

    m.def("energy_calc_msa", &energy_calc_msa, 
                    "Compute the energy of each sequence in an MSA");
    m.def("energy_calc_single_mutants", &energy_calc_single_mutants, 
                    "Compute the energy of single mutants of a sequence");
}
