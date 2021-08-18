import pathlib
import logging

# to save sequences (in bytes) to a buffer in memory 
import io

import numpy as np
import itertools 
import functools
import gzip

import Bio
from Bio import Seq, SeqIO
from Bio.Data import IUPACData


class NumAlphabetEncoder:
    """A class that contains tables that can encode strings into integers
    according to some alphabet. All conversions are done with bytes and
    np.uint8 so that they remain fast. None of the numbers can be above 256.

    >>> na = NumAlphabetEncoder(alphabet="ACTG", numbers=[0,1,2,3])
    >>> na.alpha_to_int_bytes_table[:10] 
    b'\\x00\\x01\\x02\\x03\\x04\\x05\\x06\\x07\\x08\\t'
    >>> na.alpha_to_int_dict
    {'A': 0, 'C': 1, 'T': 2, 'G': 3}
    """

    def __init__(self, alphabet, numbers=None):
        """ alphabet is a string """
        self._alphabet = None # alphabet in strings
        self._numbers = None
        self._alphabet_b = None # alphabet in bytes
        self._numbers_b = None # encoding in bytes
        self._alpha_to_int_table = None # convert letters to numbers (bytes)
        self._int_to_alpha_table = None # convert numbers to letters (bytes)

        self.init_vars(alphabet, numbers)


    def init_vars(self, alphabet, numbers=None):
        self._alphabet = alphabet
        if numbers is None:
            numbers = range(len(self._alphabet))
        self._numbers = list(numbers)

        # check that alphabet and numbers have the same length
        if len(self._alphabet) != len(self._numbers):
            raise ValueError("Alphabet and numbers must have the same "
                    "length. Pass in numbers=None to encode with consecutive "
                    "numbers") 

        # check that numbers are between 0 and 255 so that we can encode
        # into bytes
        if ((np.array(self._numbers) > 255).any() or 
            (np.array(self._numbers) < 0).any()):
            raise ValueError("All numbers have to be between 0 and 255")

        self._alphabet_b = self._alphabet.encode("ASCII")
        self._numbers_b = np.array(self._numbers, dtype=np.uint8).tobytes()
        self._alpha_to_int_bytes_table = bytes.maketrans(self._alphabet_b, 
                                                    self._numbers_b)
        self._int_to_alpha_bytes_table = bytes.maketrans(self._numbers_b, 
                                            self._alphabet_b)
    
    @property
    def alpha_to_int_bytes_table(self):
        return self._alpha_to_int_bytes_table

    @property
    def int_to_alpha_bytes_table(self):
        return self._int_to_alpha_bytes_table

    @property
    def alpha_to_int_dict(self):
        return dict(zip(self._alphabet, self._numbers))

    @property
    def int_to_alpha_dict(self):
        return dict(zip(self._numbers, self._alphabet))


# We can change the DEFAULT_ENCODER by using its init_vars method
DEFAULT_ENCODER = NumAlphabetEncoder(IUPACData.protein_letters + "-")

## create a mapping for non-degenerate codons
## This will be used for one-hot encoding the sequences
#codon_table = Bio.Data.CodonTable.standard_dna_table
#codon_map = {c:i for i, c in enumerate(
#                    sorted(codon_table.forward_table.keys()))}

def file_handle_opener(filename):
    """ Figure out what opener to use based on filename.
        For now this function chooses between gzip and regular open
    """
    opener = open # default opener

    # if filename is actually a filehandle then pass it through
    def handle_passthrough(*args, **kwargs):
        return args[0]
    if isinstance(filename, io.IOBase): # filename is a filehandle actually
        opener = handle_passthrough
    else: # check if file is gzipped
        if str(filename).endswith(".gz"):
            opener = gzip.open
    return opener

def txt_io_gen(filename):
    """ Read in an MSA from a plain text file. One sequence per line. Can be
    gzipped or not. If gzipped, then the filename should end in .gz

    Return: an iteratory over the sequences in bytes 
    """
    opener = file_handle_opener(filename)
    with opener(filename, "rb") as fh: # open in bytes
        for line in fh:
            yield line.rstrip()


def Bio_SeqIO_gen(filename, filetype="fasta"):
    """ Read in an MSA from fasta file. Can be gzipped or not. If gzipped, then
    the filename should end in .gz

    Return: an iteratory over the sequences in bytes 
    """
    opener = file_handle_opener(filename)
    with opener(filename, "rb") as fh: # open in bytes
        for seq in SeqIO.parse(fh, filetype):
            yield bytes(seq)

def guess_msa_filetype_from_filename(filename):
    """ Take in a filename and return fasta or txt depending on extensions.

    >>> guess_msa_filetype_from_filename("test.fasta.gz")
    'fasta'
    """
    suffixes = pathlib.Path(str(filename)).suffixes
    suffix = suffixes[-1]
    filetype = None
    if len(suffixes) > 1 and suffixes[-1] == ".gz":
        suffix = suffixes[-2] # handle .fasta.gz
    if suffix == ".fasta" or suffix == ".a2m":
        filetype = "fasta"
    elif suffix == ".txt" or suffix == ".text" or suffix == ".aln":
        filetype = "txt"
    else:
        err_str = f"Input MSA must have format FASTA/A2M or TEXT/ALN.\n" \
                  f"The file extension must be one of " \
                  f".fasta/.a2m/.txt/.text/.aln to reflect that. Or " \
                  f".fasta.gz/.a2m.gz/.txt.gz/.text.gz/.aln.gz if compressed." \
                  f"Found extension {suffix}"
        raise ValueError(err_str)
    return filetype

def get_msa_from_filename_iter(filename, seq_io_gen=None, start=None, 
                                                stop=None): 
    """Reads a MSA file and returns a numpy array of integers

    Args:
        filename  : Filename or filehandle of alignment file to read. If
                    a filehandle is passed in then filetype must be given.
        seq_io_gen: Shoud be txt_io_gen or SeqIO_gen or something like that. 
                    None means we should guess based on filename. We can only
                    guess between txt and fasta. 
                    If Bio_SeqIO_gen is needed with a fixed filetype then use
                    functools.partial to fix the filetpye and pass it in.
        start     : skip the first start number of seqs
        stop      : stop at this sequence. islice[start:stop]
    """
    if seq_io_gen is None: # guess based on filename
        filetype = guess_msa_filetype_from_filename(filename)
        if filetype == "fasta":
            seq_io_gen = functools.partial(Bio_SeqIO_gen, filetype="fasta")
        elif filetype == "txt":
            seq_io_gen = txt_io_gen
        else: # filetype is None:
            raise ValueError ("Filetype can only be fasta or aln. ",
                    "Pass in a specific sequence generator "
                    "for nonstandard files")
    seq_io_gen_slice = itertools.islice(seq_io_gen(filename), start, stop) 
    yield from (seq for seq in seq_io_gen_slice)
  

def get_msa_from_filename(filename, upper_case=True, 
                        num_encoder=DEFAULT_ENCODER, *args, **kwargs):
    """Reads a (plain text) fasta/aln file (can be gzipped also) and returns an
        MSA

    Takes a simple text file (ALN) which has one sequence per line. OR
    Takes a fasta file Returns 
    and MSA as a numpy array with integer encoding accoding to num_encoder

    Args:
        filename    : Filename or filename of ALN/FASTA file to read
        uppercase   : change sequences to upper case before convering to numbers
        num_encoder : any numerical encoding scheme with the same alphabet 
                        that exists in the file.
        args,kwargs : passed into get_msa_from_filename_iter

    Returns:
        A numpy array of dtpye uint8 which represents the MSA. 
        The first axis is the sequence number. The second axis is the
            residue number. 
    """
    tmp_fh = io.BytesIO() # create a place to write to in memory
    trans_table = num_encoder.alpha_to_int_bytes_table
    for line in get_msa_from_filename_iter(filename, *args, **kwargs):
        if upper_case: line = line.upper()
        tmp_fh.write(line.translate(trans_table)) # write seqs in bytes
    L = len(line) # needed to reshape array
    tmp_fh.seek(0) # rewind memory file pointer to start
    # tmp_fh owns the memory and numpy will not make a copy so this is fast
    arr = np.frombuffer(tmp_fh.read(), dtype=np.uint8) 
    arr = arr.reshape((-1, L)) 
 
    return arr


def save_numpy_int_arr_to_txt(arr, filename, num_encoder=DEFAULT_ENCODER):
    """ save a numpy 2D array of integer coded sequences (np.uint8) to a text
        file.

        arr     : 2D numpy array of dtype uint8
    """
    if arr.dtype != np.uint8:
        raise ValueError("Numpy array must have dtype uint8")
    if arr.ndim != 2:
        raise ValueError("Numpy array must be 2D")

    rev_trans_table = num_encoder.int_to_alpha_bytes_table
    tmp_w_fh = io.BytesIO()
    for row in arr:
        tmp_w_fh.write(row.tobytes().translate(rev_trans_table))
        tmp_w_fh.write(b'\n')

    tmp_w_fh.seek(0)

    opener = file_handle_opener(filename)
    with opener(filename, "wb") as fh:
        fh.write(tmp_w_fh.read()) 

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

    import doctest
    doctest.testmod()

    #stop

    dhfr_fn = "../../../../VAEs/sequence_sets/DHFR.aln.gz"
    arr = get_msa_from_filename(dhfr_fn)
    save_numpy_int_arr_to_txt(arr, "/tmp/testing.txt.gz")

    # FIXME: Add functionality to encode CODON arrays
   
