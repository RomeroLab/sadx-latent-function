import numpy as np

import Bio.Seq
from Bio.Data import IUPACData, CodonTable

class NumAlphabetEncoder:
    """A class that contains tables that can encode strings into integers
    according to some alphabet. All conversions are done with bytes and
    np.uint8 so that they remain fast. None of the numbers can be above 256.
    This works for any alphabet DNA, RNA, Amino Acids etc

    >>> na = NumAlphabetEncoder(alphabet="ACTG", numbers=[0,1,2,3])
    >>> na.alpha_to_int_bytes_table[:10] 
    b'\\x00\\x01\\x02\\x03\\x04\\x05\\x06\\x07\\x08\\t'
    >>> na.alpha_to_int_dict
    {'A': 0, 'C': 1, 'T': 2, 'G': 3}
    >>> na.int_mutate_dict.keys()
    dict_keys([0, 1, 2, 3])
    >>> na.int_mutate_dict[0] # what can indices can A mutate to?
    [1, 2, 3]
    >>> na.int_mutate_arr # mutation table for entire alphabet
    array([[1, 2, 3],
	   [0, 2, 3],
	   [0, 1, 3],
	   [0, 1, 2]], dtype=uint8)
    """

    def __init__(self, alphabet, numbers=None):
        """ alphabet is a string """
        self.alphabet = None # alphabet in strings
        self.numbers = None
        self.alphabet_b = None # alphabet in bytes
        self.numbers_b = None # encoding in bytes
        self.alpha_to_int_table = None # convert letters to numbers (bytes)
        self.int_to_alpha_table = None # convert numbers to letters (bytes)

        self.alpha_to_int_dict = None 
        self.int_to_alpha_dict = None

        # A mutate dictionary keyed on self.numbers with values being a list
        # of the other numbers. This tells us which amino acids can mutate to
        # which other amino acids.
        self.int_mutate_dict = None

        # A mutate np array of shape (N, q-1) which is a numpy version of
        # int_mutate_dict. Only defined for contiguous encodings
        self.int_mutate_arr = None

        self.init_vars_inplace(alphabet, numbers)


    def init_vars_inplace(self, alphabet, numbers=None):
        self.alphabet = alphabet
        if numbers is None:
            numbers = range(len(self.alphabet))
        self.numbers = list(numbers)

        # check that alphabet and numbers have the same length
        if len(self.alphabet) != len(self.numbers):
            raise ValueError("Alphabet and numbers must have the same "
                    "length. Pass in numbers=None to encode with consecutive "
                    "numbers") 

        # check that numbers are between 0 and 255 so that we can encode
        # into bytes
        if ((np.array(self.numbers) > 255).any() or 
            (np.array(self.numbers) < 0).any()):
            raise ValueError("All numbers have to be between 0 and 255")

        self.alphabet_b = self.alphabet.encode("ASCII")
        self.numbers_b = np.array(self.numbers, dtype=np.uint8).tobytes()
        self.alpha_to_int_bytes_table = bytes.maketrans(self.alphabet_b, 
                                                    self.numbers_b)
        self.int_to_alpha_bytes_table = bytes.maketrans(self.numbers_b, 
                                            self.alphabet_b)
        self.alpha_to_int_dict = dict(zip(self.alphabet, self.numbers))
        self.int_to_alpha_dict = dict(zip(self.numbers, self.alphabet))

        # store dictionary of which aa index mutates to which other aa index
        self.int_mutate_dict = {k:self.numbers.copy() for k in self.numbers}
        for k in self.numbers:
            self.int_mutate_dict[k].remove(k) # remove inplace
        self.int_mutate_arr = None
        if self.is_contiguous():
            self.int_mutate_arr = np.array(
                    list(self.int_mutate_dict.values()), dtype=np.uint8)


    def string_to_np(self, s):
        """ Convert string to numpy array using encoder
            >>> na = NumAlphabetEncoder(alphabet="ACTG")
            >>> na.string_to_np("GGCGACTCA")
            array([3, 3, 1, 3, 0, 1, 2, 1, 0], dtype=uint8)
        """
        return np.array([self.alpha_to_int_dict[a] for a in s], dtype=np.uint8)

    def np_to_string(self, arr):
        """ Convert string to numpy array using encoder
            >>> na = NumAlphabetEncoder(alphabet="ACTG")
            >>> na.np_to_string(np.array([3, 3, 1, 3, 0, 1, 2, 1, 0]))
            'GGCGACTCA'
        """
        return "".join(self.int_to_alpha_dict[i] for i in arr)


    def is_contiguous(self):
        """ check that encoding is contigous numbers 
            >>> na = NumAlphabetEncoder(alphabet="ACTG", numbers=[0,1,2,4])
            >>> na.is_contiguous()
            False
            >>> na = NumAlphabetEncoder(alphabet="ACTG")
            >>> na.is_contiguous()
            True
        """
        return max(self.numbers) == (len(self.numbers) - 1)

    def change_arr_encoding(self, arr, axes, arr_enc, drop_chars=""):
        """ Convert numpy array with elements in one encoding to another
            Params:
                arr        :numpy array (any dtype). eg. shape (L,q) or (L,q,L,q)
                axes       :numpy axis to change. eg axes=1 or axes=(1,3)
                arr_enc    :alphabet or NumAlphabetEncoder used to create arr
                drop_chars :remove indices in return value corresponding 
                            to some chars. eg.  drop_chars = "-"
            Return:
                Tuple (ret_arr, ret_enc). ret_arr is a numpy array in the 
                encoding ret_enc. ret_enc will be self if this encoding is
                contiguous or drop_chars is empty
        """
        if isinstance(arr_enc, str):
            arr_enc = NumAlphabetEncoder(arr_enc)
        # return encoding # ensures contigous numbering system
        ret_enc = self
        if drop_chars or (not self.is_contiguous()):
            # create a new encoding system that matches the return type
            ret_enc  = NumAlphabetEncoder(alphabet="".join(
                k for k in self.alphabet if k not in drop_chars))

        # these are the columns we should pick up and in this order
        reorder_cols = np.array([arr_enc.alpha_to_int_dict[k] for k in \
                ret_enc.alpha_to_int_dict.keys()])
        if isinstance(axes, int):
            axes = (axes,) # convert to tuple
        ret_arr = arr
        for ax in axes:
            ret_arr = ret_arr.take(reorder_cols, axis=ax)
        return (ret_arr, ret_enc)
       
        

# We can change the DEFAULT_ENCODER by using its init_vars method
DEFAULT_ENCODER = NumAlphabetEncoder(IUPACData.protein_letters + "-")
DEFAULT_ENCODER_DICT = DEFAULT_ENCODER.alpha_to_int_dict

DEFAULT_NO_GAP_ENCODER = NumAlphabetEncoder(IUPACData.protein_letters) 

DEFAULT_DNA_ENCODER = NumAlphabetEncoder(IUPACData.unambiguous_dna_letters)

# OLD Amino acid encoding (not to be used anymore)
OLD_AA_ORDER = "RKDEQNHSTCYWAILMFVPG-"


# Only codons that map to amino acids are included. Excludes the 3 stop codons
# There are 61 mappings from codons to Amino acids in the dict below
# {'AAA':0, ..., 'TTT': 60}
CODON_NUM_MAP = {c:i for i, c in enumerate(
    sorted(CodonTable.standard_dna_table.forward_table.keys()))}

# MAP codon numerical code in CODON_NUM_MAP to AA letter
# {0:'K', ..., 60:'F'}
CODON_NUM_AA_MAP = {v:str(Bio.Seq.Seq(k).translate()) for k,v in 
                                                    CODON_NUM_MAP.items()}
# Same as above but mapping CODON_NUM_MAP numerical code to DEFAULT_ENCODER's
# numerical code {0:8, ..., 60:4}
CODON_NUM_AA_NUM_MAP = {k:DEFAULT_ENCODER_DICT[v] for k,v in 
                                                    CODON_NUM_AA_MAP.items()}

if __name__ == "__main__":
    import doctest
    doctest.testmod()
