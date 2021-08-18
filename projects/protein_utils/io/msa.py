import pathlib
import logging

import numpy as np
import itertools 
import gzip

import Bio
from Bio import Seq, SeqIO

class NumEncoder:
    """A class that can encode strings into integers according to some alphabet"""

    # FIXME: Add a default encoder as a member of this module
    # allow the default encoder to be changed 

    def __init__(self, alphabet):
        self.alphabet = alphabet


#AMINO_ACIDS = np.array([aa for aa in "RKDEQNHSTCYWAILMFVPG-"], "S1")
## AAs = AMINO_ACIDS[:-1] # drop the gap character
#AAs = AMINO_ACIDS # idk why Sameer dropped the gap character so I'm not doing that
#AAs_string = AAs.tostring().decode("ascii")
#AA_L = AAs.size # alphabet size
#AA_map = {a:idx for idx,a in enumerate(AAs)} # map each amino acid to an index
## same map as above but with ascii indices
#AA_map_str = {a:idx for idx, a in enumerate(AAs_string)}
#
## create a mapping for non-degenerate codons
## This will be used for one-hot encoding the sequences
#codon_table = Bio.Data.CodonTable.standard_dna_table
#codon_map = {c:i for i, c in enumerate(
#                    sorted(codon_table.forward_table.keys()))}


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
    if suffix == ".fasta" or suffix == ".a2m":
        filetype = "fasta"
    elif suffix == ".txt" or suffix == ".text" or suffix == ".aln":
        filetype = "aln"
    else:
        err_str = f"Input MSA must have format FASTA/A2M or TEXT/ALN.\n" \
                  f"The file extension must be one of " \
                  f".fasta/.a2m/.txt/.text/.aln to reflect that. Or " \
                  f".fasta.gz/.a2m.gz/.txt.gz/.text.gz/.aln.gz if compressed." \
                  f"Found extension {suffix}"
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


if __name__ == "__main__":
    from Bio.Data import IUPACData
    from io import BytesIO


    alphabet = IUPACData.protein_letters + "-"
    int_encoding = np.arange(len(alphabet), dtype=np.uint8).tobytes()
    trans_table = bytes.maketrans(alphabet.encode("ASCII"), int_encoding)
    rev_trans_table = bytes.maketrans(int_encoding, alphabet.encode("ASCII"))

    def file_handle_opener(filename):
        """ Figure out what opener to use based on filename.
            For now this function chooses between gzip and regular open
        """
        opener = open
        if str(filename).endswith(".gz"):
            opener = gzip.open
        return opener

    def aln_iter_gen(filename):
        opener = file_handle_opener(filename)
        with opener(filename, "rb") as fh:
            for line in fh:
                yield line.rstrip()

    start=None
    stop=None
    dhfr_fn = "../../../../VAEs/sequence_sets/DHFR.aln.gz"
    tmp_fh = BytesIO()
    for line in itertools.islice(aln_iter_gen(dhfr_fn), start, stop):
        tmp_fh.write(line.upper().translate(trans_table))
    L = len(line)
    tmp_fh.seek(0)
    arr = np.frombuffer(tmp_fh.read(), dtype=np.uint8)
    arr = arr.reshape((L, -1))
    
    # add functionality to write fasta as well
    # throw exception if arr is not uint8
    tmp_w_fh = BytesIO()
    for row in arr:
        tmp_w_fh.write(row.tobytes().translate(rev_trans_table))
    tmp_w_fh.seek(0)


