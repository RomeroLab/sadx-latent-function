"""

Convert plmc binary parameter values to numpy version

NOTE: Run in evcouplings conda environment

"""
import os
import sys
import random

import pathlib
import logging

import numpy as np

import Bio
import Bio.SeqIO

import evcouplings
from evcouplings.couplings import CouplingsModel


from ..sequences.encoding import DEFAULT_ENCODER, NumAlphabetEncoder


# PLMC encoding has the gap character first
PLMC_ENCODER = NumAlphabetEncoder("-" + DEFAULT_ENCODER.alphabet[:20])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--plmc_params_filename",
                    help="plmc parameters (plmc v2 bin format)",
		    type=pathlib.Path,
                    required=True)
    parser.add_argument("-w", "--wildtype_fasta",
                    help="Wildtype sequence",
		    type=pathlib.Path,
                    default=None,
                    required=False)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    wt = None
    if args.wildtype_fasta:
        logging.info(f"Reading Wildtype sequence")
        wt = Bio.SeqIO.read(args.wildtype_fasta, format="fasta")
        
    logging.info(f"Loading couplings model")

    cm = CouplingsModel(args.plmc_params_filename)
    logging.info(f"Length of sequence in coupling model: "
                 f"{cm.L}")

    if wt:
        random_seq = "".join(random.choices(list(cm.alphabet_map.keys()),
            k=cm.L)) 
        logging.info(f"Hamiltonian of wildtype: "
                 f"{cm.hamiltonians([wt])}")
        logging.info(f"Hamiltonian of random  : "
                 f"{cm.hamiltonians([random_seq])}")


    h_ia, _ = DEFAULT_ENCODER.change_arr_encoding(cm.h_i, 
                            axes=1, arr_enc=PLMC_ENCODER)

    e_iajb, _ = DEFAULT_ENCODER.change_arr_encoding(cm.J_ij, axes=(2,3),
                    arr_enc=PLMC_ENCODER)
    # EVcouplings has (L, L, q, q) shape for the second order terms
    # swap the axes to put it in (L, q, L, q) shape which is what we need
    e_iajb = e_iajb.copy()
    e_iajb = np.moveaxis(e_iajb, 2, 1).copy()
    logging.info(f"e_iajb.shape : {e_iajb.shape} ")

    logging.info(f"Saving main effects numpy array")
    main_effects_filename = args.plmc_params_filename.with_suffix(".h_ia.npy")
    np.save(main_effects_filename, h_ia)

    logging.info(f"Saving pairwise effects numpy array")
    pairwise_effects_filename = args.plmc_params_filename.with_suffix(
                                            ".e_iajb.npy")
    np.save(pairwise_effects_filename, e_iajb)



    logging.info(f"Checking wt hamiltonians with numpy arrays")
    #
    q = len(DEFAULT_ENCODER.alphabet)
    wt_one_hot = np.eye(q)[DEFAULT_ENCODER.string_to_np(wt.seq)]
    h_ia_wt = np.einsum("ia,ia", h_ia, wt_one_hot)
    e_iajb_wt = np.einsum("ia,iajb,jb", wt_one_hot, e_iajb, wt_one_hot) / 2.
    hamiltonian_wt = h_ia_wt + e_iajb_wt
    logging.info(f"Numpy Hamiltonian WT : [[ {hamiltonian_wt:.8f} "
            f"{e_iajb_wt:.8f} {h_ia_wt:.8f} ]]")

    logging.info(f"Done")

