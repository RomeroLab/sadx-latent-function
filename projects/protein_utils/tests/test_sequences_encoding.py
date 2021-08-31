import unittest

from pathlib import Path
THIS_DIR = Path(__file__).parent

import numpy as np
import protein_utils.sequences.encoding as enc

class TestNumAlphabetEncoder(unittest.TestCase):

    def setUp(self):
        self.num_enc = enc.DEFAULT_ENCODER

    def test_change_arr_encoding(self):
        old_encoding = "RKDEQNHSTCYWAILMFVPG-"
        q = len(old_encoding)
        L = 10
        h = np.random.randint(q, size=(L,q))
        e = np.random.randint(q, size=(L, q, L, q))

        # now change the encoding to the default encoding
        ret_arr, ret_enc = self.num_enc.change_arr_encoding(
                h, arr_enc=old_encoding, axes=1)
        self.assertEqual(ret_enc, self.num_enc)
        # amino acid V is position 17 in old array
        new_v_pos = self.num_enc.alpha_to_int_dict["V"]
        self.assertEqual(h[5, 17], ret_arr[5, new_v_pos])

        # now check pairwise terms
        ret_arr, ret_enc = self.num_enc.change_arr_encoding(
                e, arr_enc=old_encoding, axes=(1,3))
        self.assertEqual(ret_enc, self.num_enc)
        # amino acid D is position 2 in old array
        new_d_pos = self.num_enc.alpha_to_int_dict["D"]
        self.assertEqual(e[5, 17, 8, 2], 
                ret_arr[5, new_v_pos, 8, new_d_pos])
        
        # now drop the gap character
        ret_arr, ret_enc = self.num_enc.change_arr_encoding(
                e, arr_enc=old_encoding, axes=(1,3), drop_chars="-")
        self.assertNotEqual(ret_enc, self.num_enc) # get a new encoding back
        self.assertTrue(ret_enc.is_contiguous())
        self.assertEqual(e[5, 17, 8, 2], 
                ret_arr[5, new_v_pos, 8, new_d_pos])

        
if __name__ == '__main__':
    unittest.main()


