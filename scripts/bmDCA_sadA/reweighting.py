"""Reweight sequences in an MSA to account for uneven sampling.

Reweight sequences in an MSA to account for unevening sampling and
phylogenetic biases. This script will take an MSA as input (in Fasta or ALN
format) and compute the weights of each sequence. The memory reqiurements and
the computational complexitity for a naive algorithm grow according to the
square of the number of sequences. This isn't a problem for MSAs with ~10k
sequences but quickly becomes a problem with MSAs of size ~100k and above.
The design goal is to be able to use it as a standalone script or as a module
and imported into other python programs.

  Typical usage example:

  weights = compute_weights_from_msa(threshold_distance)
"""

import numpy as np
import torch

from dataloader import get_msa_from_file

def compute_weights_from_aligned_msa(msa, threshold, device):
    """Computes weights from an msa using threshold to determine neighbors.

    Counts the number of neighbors around each sequence using the threshold
    as a cutoff to determine what a neighbor is. The weight for each sequence
    is the reciprocal of the count of how many neighbors it has.

    Args:
        msa: A numpy array of dtype S1 with 2 dimensions and each sequence as
            a row.
        threshold: A float that determines the cutoff at which a sequence
            with distance similarity to another sequence is considered a
            neighbor. Hamming distance >= 80% would have 0.8 for this value.
        device: pytorch device to store the Tensors. This function can take
            very long to compute on a CPU. For ~100k sequences we estimate
            around 20 minutes. So hopefully device is "cuda:0".

    Returns:
        A numpy 1D array of weights which has the same number of elements as
        the the number of sequences (rows or first dimension) in the msa that
        is passed in. 
    """
    with torch.no_grad():
        N, L = msa.shape # number of sequences, length of protein
        msa_int = torch.ByteTensor(msa.view(np.uint8))

        # Compute integer threshold
        distance_to_threshold = threshold 
        epsilon = 1e-6 # to avoid rounding issues (unlikely)
        distance_from_threshold_int = int((1 - distance_to_threshold)*L + epsilon)

        # Create a torch scalar for the threshold
        torch_threshold = torch.ShortTensor(1)
        torch_threshold[0] = distance_from_threshold_int
        torch_threshold = torch_threshold.to(device)

        msa_int = msa_int.to(device) # move to device
        torch_seqs = torch.unbind(msa_int, dim=0) # split into separate sequences

        def count_neighbors(torch_seq):
            """Count the neighbors of torch_seq.
            This is an inline function that uses msa_int and torch_threshold"""
            dist_from_seq = (torch_seq != msa_int).sum(dim=1, dtype=torch.short)
            threshold_count = (dist_from_seq <=
                    torch_threshold).sum(dtype=torch.short) 
            return threshold_count

        neighbors_count = torch.stack(tuple(count_neighbors(torch_seq) for
            torch_seq in torch_seqs))

        weights = 1 / neighbors_count.float()
    weights = weights.data.cpu().numpy()

    # normalize weights to sum to length of MSA
    weights = (weights / weights.sum()) * N
    return weights




if __name__ == "__main__":
    import time
    import argparse
    import read_config

    parser = argparse.ArgumentParser()
    parser.add_argument("config_filename",
                    help="input config file in yaml format")
    args = parser.parse_args()

    config = read_config.Config(args.config_filename)

    msa = get_msa_from_file(config.aligned_msa_fullpath, as_numpy=True)

    start_time = time.time()
    weights = compute_weights_from_aligned_msa(msa, 
            threshold=config.reweighting_threshold,
            device=config.device)
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

    np.save(config.weights_fullpath, weights, allow_pickle=False)

