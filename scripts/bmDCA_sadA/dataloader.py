import pathlib
import logging

import numpy as np
import itertools 
import gzip

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import Bio
from Bio import Seq, SeqIO

AMINO_ACIDS = np.array([aa for aa in "RKDEQNHSTCYWAILMFVPG-"], "S1")
# AAs = AMINO_ACIDS[:-1] # drop the gap character
AAs = AMINO_ACIDS # idk why Sameer dropped the gap character so I'm not doing that
AAs_string = AAs.tostring().decode("ascii")
AA_L = AAs.size # alphabet size
AA_map = {a:idx for idx,a in enumerate(AAs)} # map each amino acid to an index
# same map as above but with ascii indices
AA_map_str = {a:idx for idx, a in enumerate(AAs_string)}

# create a mapping for non-degenerate codons
# This will be used for one-hot encoding the sequences
codon_table = Bio.Data.CodonTable.standard_dna_table
codon_map = {c:i for i, c in enumerate(
                    sorted(codon_table.forward_table.keys()))}


def get_msa_from_filename_iter(filename, filetype, size_limit):
    """Reads a fasta file and returns a string iterator  

    Args:
        filename  : Filename or filehandle of fasta file to read
        filetype  : "fasta" or "aln"
    """
    opener = open
    if str(filename).endswith(".gz"):
        opener = gzip.open
    with opener(filename, "rt") as fh:
        seq_io_gen = None  # raw string iterator over sequences
        if filetype == "fasta":
            seq_io_gen = (str(r.seq) for r in SeqIO.parse(fh, "fasta")) 
        elif filetype == "aln":
            seq_io_gen = (line.strip() for line in fh)
        if seq_io_gen is None:
            raise ValueError ("filetype can only be fasta or aln")
        # Read only size_limit elements of the generator
        # if size_limit is None then we will read in everything
        seq_io_gen_slice = itertools.islice(seq_io_gen, size_limit) 
        # Here we can return a generator expression because SeqIO.parse
        # is handling the file handle
        yield from (seq.upper() for seq in seq_io_gen_slice)
  

def get_msa_from_filename(filename, filetype, size_limit=None, 
                            as_numpy=True, as_iter=False):
    """Reads a (plain text) fasta/aln file (can be gzipped also) and returns an
        MSA

    Takes a simple text file (ALN) which has one sequence per line. OR
    Takes a fasta file Returns 
    and MSA as a numpy array or as a list of Bio.Seq sequences.

    Args:
        filename    : Filename or filename of ALN/FASTA file to read
        filetype    : "fasta" or "aln"
        size_limit  : Return upto size_limit sequences
        as_numpy    : return numpy byte array instead of list of seqs
        as_iter     : return the raw string iterator instead 

    Returns:
        if as_iter is True:
            A raw string iterator
        if as_numpy is False:
            A list of sequences in Bio.Seq format
        if as_numpy is True:
            A numpy byte array of dtpye S1 which represents the MSA. 
            The first axis is the sequence number. The second axis is the
            residue number. 
    """
    seqs = get_msa_from_filename_iter(filename, filetype, size_limit)
    ret = None
    if as_iter: # return the raw iterator
        ret = seqs
    elif as_numpy:
        # convert to lists of lists for easy numpy conversion to 2D array
        ret = np.array([list(s) for s in seqs], dtype="|S1")
    else:
        ret = [Seq.Seq(s) for s in seqs]
    return ret


def get_msa_from_file(msa_file, size_limit=None, as_numpy=True, as_iter=False):
    """
        Read in the filename and call the right function to read in the MSA
        by looking at the extension
    """
    suffixes = pathlib.Path(str(msa_file)).suffixes
    suffix = suffixes[-1]
    filetype = None
    if len(suffixes) > 1 and suffixes[-1] == ".gz":
        suffix = suffixes[-2] # handle .fasta.gz
    if suffix == ".fasta" or suffix == ".a2m" or suffix == ".afa":
        filetype = "fasta"
    elif suffix == ".txt" or suffix == ".text" or suffix == ".aln":
        filetype = "aln"
    else:
        err_str = f"Input MSA must have format FASTA/A2M or TEXT/ALN.\n" \
                  f"The file extension must be one of " \
                  f".fasta/.a2m/.afa/.txt/.text/.aln to reflect that. Or " \
                  f".fasta.gz/.a2m.gz/.afa.gz/.txt.gz/.text.gz/.aln.gz " \
                  " if compressed. Found extension {suffix}"
        raise ValueError(err_str)
    return get_msa_from_filename(msa_file, filetype, size_limit=size_limit,
            as_numpy=as_numpy, as_iter=as_iter)

def get_codon_msa_as_int_array(filename, codon_map):
    """
        Returns an CODON msa MSA as a two dimension numpy array
        (N, L) where N = Number of sequences in the MSA
                     L = Length of the protein (# of Amino Acid Residues)
               and each value in this array is the value of codon_map

        `codon_map`: a dictionary mapping codons (as strings of length 3)
                     to integer values
    """
    seq_iter = get_msa_from_file(filename, as_iter=True)

    def codon_seq_to_int_list(seq): 
        return [codon_map[seq[3*i:(3*i+3)]] for i in range(len(seq)//3)]
    
    return np.array([codon_seq_to_int_list(seq) for seq in seq_iter], 
                        dtype=np.uint8) 


class MSADataset(Dataset):
    '''Reads an MSA and converts to pytorch dataset'''

    def __init__(self, msa_file, size_limit=None, weights=None, 
                 transform=None,
                 convert_unknown_aa_to_gap=False):
        self.raw_data = self.get_raw_data(msa_file, size_limit)
        self.transform = transform
        self.AA_enc = self.get_encoding_dict()
        if convert_unknown_aa_to_gap:
            gap_enc = self.AA_enc['-']
            for k in "BJOUXZ":  # set these amino acids to gap
                self.AA_enc[k] = gap_enc
        N = self.__len__()
        if weights is None:
            logging.info("Weights are not specified in dataloader. "
                         "Setting equal weights.")
            logging.info("... [NOTE] when training the model, Check to see if "
                         "sampling is weighted")
            self.weights = np.ones(N, dtype=np.float32);
        elif isinstance(weights, np.ndarray):
            self.weights = weights
        else:
            logging.warning("Weights param should not be a filename")
            self.weights = np.array(np.load(weights)).astype(np.float32).squeeze()
        assert(self.weights.shape[0]==N)

    def get_raw_data(self, msa_file, size_limit):
        return get_msa_from_file(msa_file, size_limit=size_limit, as_numpy=False)

    def get_used_encoding_dict():
        return self.AA_enc.copy()

    @staticmethod
    def get_encoding_dict(): # default encoding dictionary
        return AA_map_str.copy()

    def __len__(self):
        return len(self.raw_data)

    def protein_to_tensor(self, prot_str):
        # Integer representation of the protein
        prot_int = [self.AA_enc[p] for p in prot_str]
        sample = torch.LongTensor(prot_int)
        if self.transform:
            sample = self.transform(sample)
        return (sample)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = (self.protein_to_tensor(self.raw_data[idx]), self.weights[idx])
        return sample

class OneHotTransform:
    
    def __init__(self, num_labels, to_float=True, flatten=True):
        self.num_labels = num_labels
        self.to_float = to_float
        self.flatten = flatten

    def __call__(self, sample):
        ret = F.one_hot(sample, self.num_labels)
        if self.to_float:
            ret = ret.float()
        if self.flatten:
            ret = ret.flatten()
        return ret

