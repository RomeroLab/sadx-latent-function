import unittest

from pathlib import Path
THIS_DIR = Path(__file__).parent

import numpy as np
import tempfile # for file writing
import gzip
import protein_utils.io.msa as io_msa

class TestFastaRead(unittest.TestCase):

    def setUp(self):
        self.arr = io_msa.get_msa_from_filename(
                            THIS_DIR / "io/test_AV.fasta")

    def test_fasta_read(self):
        self.assertIsNotNone(self.arr)
        self.assertEqual(self.arr.shape, (10,10))
        self.assertEqual(self.arr.dtype, np.uint8)
        self.assertEqual(self.arr[0,0], 0) # A
        self.assertEqual(self.arr[9,9], 20) # -
        self.assertEqual(self.arr[9,1], 17) # V
        
    def test_gz_fasta_read(self):
        arr_gz = io_msa.get_msa_from_filename(
                        THIS_DIR / "io/test_AV.fasta.gz")
        self.assertTrue((self.arr == arr_gz).all())


class TestDHFRAARead(unittest.TestCase):

    def setUp(self):
        self.arr = io_msa.get_msa_from_filename(
                            THIS_DIR / "io/DHFR_Gen15_head.txt")
        with open(THIS_DIR / "io/DHFR_Gen15_head.txt", "rt") as fh_comp:
            self.arr_text = fh_comp.read()

    def test_dhfr_read(self):
        self.assertIsNotNone(self.arr)
        self.assertEqual(self.arr.shape, (10,186))
        self.assertEqual(self.arr.dtype, np.uint8)
        self.assertEqual(self.arr[0,0], 17) #V
        self.assertEqual(self.arr[9,186-1], 2) #D
        
    def test_gz_dhfr_read(self):
        arr_gz = io_msa.get_msa_from_filename(
                            THIS_DIR / "io/DHFR_Gen15_head.txt.gz")
        self.assertTrue((self.arr == arr_gz).all())

    def test_file_save_txt(self):
        with tempfile.NamedTemporaryFile(suffix=".txt") as fh:
            filename = fh.name
            io_msa.save_numpy_int_arr_to_txt(self.arr, filename)
            fh.seek(0)
            with open(filename, "rt") as fh_read:
                written_text = fh_read.read()
        self.assertEqual(written_text, self.arr_text)
                
    def test_file_save_gz(self):
        with tempfile.NamedTemporaryFile(suffix=".aln.gz") as fh:
            filename = fh.name
            io_msa.save_numpy_int_arr_to_txt(self.arr, filename)
            fh.seek(0)
            with gzip.open(filename, "rt") as fh_read:
                written_text = fh_read.read()
        self.assertEqual(written_text, self.arr_text)
 

class TestDHFRNTSRead(unittest.TestCase):

    def setUp(self):
        self.arr = io_msa.get_msa_from_filename(
                THIS_DIR / "io/DHFR_Gen15_nts_head.txt", 
                num_encoder = io_msa.DEFAULT_DNA_ENCODER )

    def test_dhfr_read(self):
        self.assertIsNotNone(self.arr)
        self.assertEqual(self.arr.shape, (10,186*3))
        self.assertEqual(self.arr.dtype, np.uint8)
        self.assertEqual(self.arr[0,0], 0) #G
        self.assertEqual(self.arr[9,186*3-1], 2) #T
        
    def test_gz_dhfr_read(self):
        arr_gz = io_msa.get_msa_from_filename(
                THIS_DIR / "io/DHFR_Gen15_nts_head.txt.gz", 
                num_encoder = io_msa.DEFAULT_DNA_ENCODER )
        self.assertEqual(arr_gz.dtype, np.uint8)
        self.assertEqual(arr_gz.shape, self.arr.shape)
        self.assertTrue((self.arr == arr_gz).all())


class TestDHFRCodonRead(unittest.TestCase):

    def setUp(self):
        self.arr = io_msa.get_codon_msa_as_int_array(
                THIS_DIR / "io/DHFR_Gen15_nts_head.txt") 

    def test_dhfr_codon_read(self):
        self.assertIsNotNone(self.arr)
        self.assertEqual(self.arr.shape, (10,186))
        self.assertEqual(self.arr.dtype, np.uint8)
        self.assertEqual(self.arr[0,0], 47) #GTT
        self.assertEqual(self.arr[10-1,186-1], 35) #GAT
        self.assertEqual(self.arr[10-1,186-2], 0) #AAA

    def test_codon_to_aa(self):
        # load the aa file from the disk
        arr_aa = io_msa.get_msa_from_filename(
                THIS_DIR / "io/DHFR_Gen15_head.txt") 
        # convert the nts file to the aa file
        arr_nts_aa = io_msa.translate_np(self.arr, io_msa.CODON_NUM_AA_NUM_MAP)
        self.assertTrue((arr_aa == arr_nts_aa).all())


if __name__ == '__main__':
    unittest.main()


