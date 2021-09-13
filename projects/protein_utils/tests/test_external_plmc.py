import unittest
import math

import protein_utils.external.plmc as plmc

from pathlib import Path
THIS_DIR = Path(__file__).parent

class TestExternalPlmc(unittest.TestCase):

    def setUp(self):
        self.filename = THIS_DIR / "external/plmc_test.gz"

    def test_read_nosort(self):
        arr = plmc.read_plmc_coupling_scores(self.filename, sort=False)
        # should be 0 - 1 instead of 1 - 2

        # size should be L choose 2
        self.assertEqual(arr.size, math.comb(186, 2))

        # the first value should be 0, 1 and not 1, 2
        a0 = arr[0]
        self.assertEqual(a0["idx1"], 0)
        self.assertEqual(a0["idx2"], 1)
        self.assertAlmostEqual(a0["score"], 0.000860)

    def test_read_sort(self):
        arr = plmc.read_plmc_coupling_scores(self.filename, sort=True)
        a0 = arr[0]

        # zcat plmc_test.gz | awk '{print $6}' | sort -rn | head -n 1
        # zcat plmc_test.gz | grep 0.012860 
        self.assertEqual(a0["idx1"], 123)
        self.assertEqual(a0["idx2"], 168)
        self.assertAlmostEqual(a0["score"], 0.012860)





if __name__ == '__main__':
    unittest.main()
    self = TestExternalPlmc()
    self.setUp()
    arr = plmc.read_plmc_coupling_scores(self.filename, sort=False)

