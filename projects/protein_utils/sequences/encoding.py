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
    """

    def __init__(self, alphabet, numbers=None):
        """ alphabet is a string """
        self._alphabet = None # alphabet in strings
        self._numbers = None
        self._alphabet_b = None # alphabet in bytes
        self._numbers_b = None # encoding in bytes
        self._alpha_to_int_table = None # convert letters to numbers (bytes)
        self._int_to_alpha_table = None # convert numbers to letters (bytes)

        self._alpha_to_int_dict = None 
        self._int_to_alpha_dict = None

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
        self._alpha_to_int_dict = dict(zip(self._alphabet, self._numbers))
        self._int_to_alpha_dict = dict(zip(self._numbers, self._alphabet))

    def string_to_np(self, s):
        return np.array([self.alpha_to_int_dict[a] for a in s], dtype=np.uint8)

    @property
    def alpha_to_int_bytes_table(self):
        return self._alpha_to_int_bytes_table

    @property
    def int_to_alpha_bytes_table(self):
        return self._int_to_alpha_bytes_table

    @property
    def alpha_to_int_dict(self):
        return self._alpha_to_int_dict

    @property
    def int_to_alpha_dict(self):
        return self._int_to_alpha_dict


# We can change the DEFAULT_ENCODER by using its init_vars method
DEFAULT_ENCODER = NumAlphabetEncoder(IUPACData.protein_letters + "-")
DEFAULT_DNA_ENCODER = NumAlphabetEncoder(IUPACData.unambiguous_dna_letters)

DEFAULT_ENCODER_DICT = DEFAULT_ENCODER.alpha_to_int_dict

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
