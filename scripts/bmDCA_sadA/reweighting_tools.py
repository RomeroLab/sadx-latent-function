"""Standalone script to reweight sequences.

  Typical usage example: (from the command line)
	python reweighting_tools.py \
		-i ../sequence_sets/cmx_aligned_blank_90.fasta \
		-t 0.8 \
		-o ../working/cmx_aligned_blank_90_weights.npy

"""

import numpy as np
import reweighting

if __name__ == "__main__":
    import time
    import argparse
    import read_config
 
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--msa_filename",
                    help="input msa in ALN or FASTA format",
                    required=True)
    parser.add_argument("-t", "--theta",
                    help="Thresholding Parameter (default 0.8)", default=0.8,
                    type=float)
    parser.add_argument("-o", "--output_filename",
                    help="Output filename for weights", default="weights.npy")
    parser.add_argument("-d", "--device",
                    help="Device to use", default="")
    args = parser.parse_args()

    msa = reweighting.get_msa_from_file(args.msa_filename, as_numpy=True)

    if not args.device:
        device = read_config.get_best_device()
    else:
        device = args.device

    start_time = time.time()
    weights = reweighting.compute_weights_from_aligned_msa(msa, 
            threshold=args.theta,
            device = device
            )
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

    np.save(args.output_filename, weights, allow_pickle=False)

