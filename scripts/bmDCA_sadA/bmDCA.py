import numpy as np

from fileio import file_handle_opener
from encoding import DEFAULT_ENCODER

# same encoder as DEFAULT_ENCODER except the gap is at the beginning
bmDCA_encoder_string = "-ACDEFGHIKLMNPQRSTVWY"

def read_bmDCA_params(filename, enc=DEFAULT_ENCODER):
    interaction_effects = dict()
    main_effects = dict()
    opener = file_handle_opener(filename)
    with opener(filename, 'rt') as fh:
        for line in fh:
            l = line.strip().split()
            if l[0] == "J":
                # this format is i,j,a,b
                interaction_effects[tuple(map(int, l[1:5]))] = float(l[5])
            elif l[0] == "h":
                main_effects[tuple(map(int, l[1:3]))] = float(l[3])
    L = max(i for i, _ in main_effects) + 1
    q = max(q for _, q in main_effects) + 1
    e_i_a_j_b = np.zeros((L,q,L,q), dtype=float)
    h_i_a = np.zeros((L,q), dtype=float)

    for (i,j,a,b), val in interaction_effects.items():
        # convert i,j,a,b to i,a,j,b
        e_i_a_j_b[i,a,j,b] = val
    for (i,a), val in main_effects.items():
        h_i_a[i,a] = val

    # release memory before creating another array
    del interaction_effects, main_effects

    ret_e, _ =  DEFAULT_ENCODER.change_arr_encoding(e_i_a_j_b, axes=(1,3), 
                    arr_enc=bmDCA_encoder_string)  
    ret_h, _ =  DEFAULT_ENCODER.change_arr_encoding(h_i_a, axes=1, 
                    arr_enc=bmDCA_encoder_string)  
    return ret_e, ret_h


    
if __name__ == "__main__":
    import tqdm
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_params_filename",
                    help="input params from bmDCA in txt or txt.gz format",
                    required=True)
    parser.add_argument("-o", "--output_filename",
                    help="Output files named output_filename_e_i_a_j_b"
                         " and output_filename_h_i_a are saved",
                    required=True)
    args = parser.parse_args()

    e_i_a_j_b, h_i_a = read_bmDCA_params(args.input_params_filename)
    np.save(args.output_filename + "_e_i_a_j_b.npy", e_i_a_j_b)
    np.save(args.output_filename + "_h_i_a.npy", h_i_a)



