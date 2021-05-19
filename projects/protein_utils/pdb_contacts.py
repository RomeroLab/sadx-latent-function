""" Extract distance_matrix or contacts from a pdb file """

import numpy as np
import logging

import Bio
import Bio.PDB


def extract_chain(filename, model_num, chain_id):
    """Gets a chain from a pdb filename """
    parser = Bio.PDB.PDBParser()
    structure = parser.get_structure("generic", args.pdb_filename)
    logging.info("Loaded structure from file : %s", args.pdb_filename)

    logging.info("Number of models: %s", len(structure))
    model = structure[args.model_num]
    logging.info("Selected model : %s", args.model_num)

    logging.info("Number of chains in model %s: %s", args.model_num, len(model))
    chain = model[args.chain_id]
    logging.info("Selected chain : %s", args.chain_id)

    return chain


def get_cb_coordinates(chain):
    """Get the CB coordinates for each residue in a chain
        For GLY we use the CA coordinate instead"""

    residue_coords = [] # list of residue coordinates 

    for residue in chain:
        residue_id = residue.get_id()
        if residue_id[0] == " ": # it isn't a hetero-residue or water
            if residue.get_resname() == "GLY":
                target_atom = residue['CA']
            else:
                target_atom = residue['CB']
            residue_coords.append(target_atom)

    logging.info("Number of residue coordinates picked: %s", 
                    len(residue_coords))

    return residue_coords

def get_dist_mat_from_coords(residue_coords):
    """ Convert a single coordinate for each residue into a distance map """
    
    L = len(residue_coords)
    dist_mat = np.zeros((L,L), dtype=np.float)

    for seq_id1, c1 in enumerate(residue_coords):
        for seq_id2, c2 in enumerate(residue_coords):
            if seq_id2 > seq_id1:
                continue
            else:
                dist_mat[seq_id1, seq_id2] = c2 - c1
                dist_mat[seq_id2, seq_id1] = dist_mat[seq_id1, seq_id2]

    return dist_mat

def get_5_8_contact_format(dist_mat, min_residue_sep):
    """ Mark "contacts" on a matrix of distances
        distances less than 8A are marked 8
        distances less than 5A are marked 5
        All other distances are marked as 0
        if residue_sequence_seperation is less than min_residue_sep
            then it overrides all distances and is marked as 0
    """
    contact_map = np.zeros(dist_mat.shape, dtype=np.int)
    contact_map[dist_mat < 8] = 8
    contact_map[dist_mat < 5] = 5 # the 5s will overwrite the 8s
    c_x, c_y = np.indices(contact_map.shape)
    close_residues_mask = (np.abs(c_x - c_y) < min_residue_sep)
    contact_map[close_residues_mask] = 0

    return contact_map


if __name__ == "__main__":
    import time
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--pdb_filename",
                        help="input PDB file", required=True)
    parser.add_argument("-m", "--model_num",
                        help="model number (0-start indexing) ",
                        default=0, type=int, required=False)
    parser.add_argument("-c", "--chain_id",
                        help="Chain ID", default="A", required=False)
    parser.add_argument("-l", "--log_level",
                        help="Logging level", default="INFO", required=False)
    parser.add_argument("-d", "--distance_matrix_filename",
                        help="Output filename for distance matrix in npy fmt",
                        default=None, required=False)
    parser.add_argument("--contact_5_8_filename",
                        help="Contact map in distance 5 and distance 8 format",
                        default=None, required=False)
    parser.add_argument("--contact_min_sep",
                        help="Minimum separation for contact. default >= 5",
                        default=5, required=False)

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    p = pathlib.Path(args.pdb_filename)
    chain = extract_chain(filename = p, model_num = args.model_num,
                            chain_id = args.chain_id)

    residue_coords = get_cb_coordinates(chain)

    dist_mat = get_dist_mat_from_coords(residue_coords)

    dist_mat_filename = args.distance_matrix_filename
    if dist_mat_filename is None:
        dist_mat_filename = p.parent / (p.stem + "_distances.npy")
    np.save(dist_mat_filename, dist_mat)

    contact_5_8_filename = args.contact_5_8_filename
    if contact_5_8_filename is None:
        contact_5_8_filename = p.parent / (p.stem + "_5_8_contacts.npy")
    contact_5_8 = get_5_8_contact_format(dist_mat, 
                        min_residue_sep = args.contact_min_sep)
    np.save(contact_5_8_filename, contact_5_8)

