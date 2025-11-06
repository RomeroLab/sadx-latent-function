"""Read/Write MSA files and return numerically encoded numpy arrays"""

import itertools 
import functools

import io
import pathlib
#import logging

import numpy as np

import Bio
import Bio.SeqIO
 
from . import encoding as enc
from ..utils.fileio import file_handle_opener, txt_io_gen


def Bio_SeqIO_gen(filename, filetype="fasta"):
    """ Read in an MSA from fasta file. Can be gzipped or not. If gzipped, then
    the filename should end in .gz

    Return: an iteratory over the sequences in bytes 
    """
    opener = file_handle_opener(filename)
    with opener(filename, "rt") as fh: # open in text
        for seq_record in Bio.SeqIO.parse(fh, filetype):
            # Biopython is now storing all seqs in bytes from version 1.9
            # so we convert using the bytes function. This only works in
            # Biopython>=1.80. For earlier versions, change this to "encode"
            yield bytes(seq_record.seq) # convert to bytes 


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
    if suffix == ".fasta" or suffix == ".a2m" or suffix == ".afa":
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
                        num_encoder=enc.DEFAULT_ENCODER, *args, **kwargs):
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


def save_numpy_int_arr_to_txt(arr, filename, num_encoder=enc.DEFAULT_ENCODER):
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
    tmp_w_fh.seek(0) # rewind to start
    opener = file_handle_opener(filename)
    with opener(filename, "wb") as fh:
        fh.write(tmp_w_fh.read())  # why can't we use getbuffer() here?

def codon_str_to_int(c, enc_d=enc.DEFAULT_DNA_ENCODER.alpha_to_int_dict):
    """ Convert codon to a number according to the DNA encoding

    >>> codon_str_to_int("GCT")
    14
    """
    if len(c) != 3:
        raise ValueError("Codon length must be 3")
    return enc_d[c[0]]*16 + enc_d[c[1]]*4 + enc_d[c[2]]*1

def bytes_trans_table_from_dict(d):
    """ convert a dictionary mapping integers to other integers (0 <= i < 255)
        into a translate table """
    return bytes.maketrans(bytearray(d.keys()), bytearray(d.values()))

def translate_np(arr, trans_dict=None, trans_table=None):
    """ Convert a numpy array of uint8s with one integer encoding to another.

    An slower altnerative to using this function is to use
    np.vectorize(trans_dict.get)(arr) 

    >>> translate_np(np.array([[0,2],[1,3]], dtype=np.uint8), {0:3, 1:5, 3:1})
    array([[3, 2],
           [5, 1]], dtype=uint8)

    Params: 
        arr         : numpy array dtpye np.uint8
        trans_dict  : a dict of integers mapping to other integers 
                        (ignored if trans_table is provided)
        trans_table : a byte translate table (will be created from
                        trans_dict if not provided)
    """
    if arr.dtype != np.uint8:
        raise ValueError("Numpy array must have dtpye uint8")
    if trans_table is None:
        trans_table = bytes_trans_table_from_dict(trans_dict)
    arr_shape = arr.shape
    arr = np.frombuffer(arr.tobytes().translate(trans_table), dtype=np.uint8)
    arr.shape = arr_shape
    return arr

def get_codon_msa_as_int_array(filename, codon_map=enc.CODON_NUM_MAP, *args, 
                                                            **kwargs):
    """
        Returns an CODON msa MSA as a two dimension numpy array
        (N, L) where N = Number of sequences in the MSA
                     L = Length of the protein (# of Amino Acid Residues)
               and each value in this array is the value of codon_map

        `codon_map`: a dictionary mapping codons (as strings of length 3)
                     to integer values
    """
    arr = get_msa_from_filename(filename, num_encoder=enc.DEFAULT_DNA_ENCODER,
                                *args, **kwargs)
    N, L = arr.shape
    if L%3: # L needs to be perfectly divisible by 3
        raise ValueError(f"Length %{L} is not divisible by 3")
    arr3 = arr.reshape(N, int(L/3), 3)

    # convert each set of 3 codons to a codon index
    codon_base4 = np.array([16,4,1], dtype=np.uint8)
    arrc = np.einsum("nic,c->ni", arr3, codon_base4, dtype=np.uint8)

    # the values of CODON_NUM_MAP are 0-60
    # this mapping maps the codon index 0-63 to a value of CODON_NUM_MAP 0-60
    # the three stop codons are not in this mapping
    # codon_str_to_int does the same thing as np.einsum above
    packed_codon_map = {codon_str_to_int(k):v for k,v in 
                                enc.CODON_NUM_MAP.items()}

    return translate_np(arrc, packed_codon_map)
 



if __name__ == "__main__":
    import doctest
    doctest.testmod()

    import tempfile


    # Read DHFR datasets AAs and Codons as numpy arrays encoded as integers

    # MSA with AA's
    dhfr_fn = "../tests/sequences/DHFR_Gen15_head.txt.gz"
    arr = get_msa_from_filename(dhfr_fn)
    with tempfile.NamedTemporaryFile(suffix=".txt.gz") as fout:
        save_numpy_int_arr_to_txt(arr, fout.name)

    # MSA in codons
    codon_fn = "../tests/sequences/DHFR_Gen15_nts_head.txt.gz"
    arr = get_codon_msa_as_int_array(codon_fn)

    # MSA in fasta format
    test_fn = "../tests/sequences/test_AV.fasta"
    arr = get_msa_from_filename(test_fn)
    
 

