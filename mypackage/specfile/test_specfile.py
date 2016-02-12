"""Tests for specfile wrapper"""
import unittest

from specfile import *

sf = SpecFile("t.dat")
scan2_1 = sf["2.1"]
scan2 = sf[2]

class TestSpecFile(unittest.TestCase):
    def test_open(self):
        self.assertIsInstance(SpecFile("t.dat"), SpecFile)
        self.assertEqual(SpecFile("t.dat").get_error_string(), "OK ( SpecFile )")
        
    def test_number_of_scans(self):
        self.assertEqual(100, len(sf))
        
    def test_list_of_scan_indices(self):
        self.assertEqual(len(sf.list()), 100)
        self.assertEqual(max(sf.list()), 100)
        self.assertEqual(min(sf.list()), 1)
        
    def test_getitem(self):
        self.assertIsInstance(sf[2], Scan)
        self.assertIsInstance(sf["2.1"], Scan)
        self.assertEqual(sf["3"].index, sf[3].index)   # should we allow int(str) indexing?
        # int out of range
        with self.assertRaisesRegexp(IndexError, 'Scan index must be in ran'):
            scan108 = sf[108]
        # float indexing not allowed
        with self.assertRaisesRegexp(IndexError, 'The scan identification k'):
            scan1 = sf[1.2]
        # non existant scan with "N.M" indexing 
        with self.assertRaisesRegexp(IndexError, 'Scan not found error \( S'):
            scan3_2 = sf["3.2"]
        
    def test_scan_headers(self):
        self.assertEqual(scan2.header_dict['S'], scan2_1.header_dict['S'])
        self.assertEqual(scan2.header_dict['S'], "He")
        self.assertNotEqual(sf["3.1"].header_dict['S'], scan2.header_dict['S'])
        self.assertEqual(scan2.header_dict['N'], 7)
        
    def test_scan_labels(self):
        self.assertEqual(scan2.header_dict['L'], 
                         ['PhotonEnergy[keV]', 'Rayleigh(coherent)[cm2/g]', 'Compton(incoherent)[cm2/g]', 'CoherentPlusIncoherent[cm2/g]', 'Photoelectric[cm2/g]', 'PairProduction[cm2/g]', 'TotalCrossSection[cm2/g]'])


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestSpecFile))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
