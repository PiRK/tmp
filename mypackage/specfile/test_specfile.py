"""Tests for specfile wrapper"""
import unittest

from specfile import *

filename1 = "t.dat"
sf1 = SpecFile(filename1)
scan2_1 = sf1["2.1"]
scan2 = sf1[1]

sf2 = SpecFile("Dry_run_feb_2016.dat")
scan_w_motors = sf2[1]


class TestSpecFile(unittest.TestCase):
    def test_open(self):
        self.assertIsInstance(SpecFile(filename1), SpecFile)
        with self.assertRaises(IOError):
            sf2 = SpecFile("doesnt_exist.dat")
        
    def test_number_of_scans(self):
        self.assertEqual(100, len(sf1))
        
    def test_list_of_scan_indices(self):
        self.assertEqual(len(sf1.list()), 100)
        self.assertEqual(max(sf1.list()), 99)
        self.assertEqual(min(sf1.list()), 0)
        
    def test_index_number_order(self):
        self.assertEqual(sf1.index(3, 1), 2)
        self.assertEqual(sf1.number(4), 5)
        self.assertEqual(sf1.order(4), 1) 
        with self.assertRaises(IndexError):
            sf1.index(3, 2)
        
    def test_getitem(self):
        self.assertIsInstance(sf1[2], Scan)
        self.assertIsInstance(sf1["2.1"], Scan)
        # int out of range
        with self.assertRaisesRegexp(IndexError, 'Scan index must be in ran'):
            scan108 = sf1[107]
        # float indexing not allowed
        with self.assertRaisesRegexp(TypeError, 'The scan identification k'):
            scan1 = sf1[1.2]
        # non existant scan with "N.M" indexing 
        with self.assertRaises(KeyError):
            scan3_2 = sf1["3.2"]
                
            
class TestScan(unittest.TestCase):
    def test_scan_headers(self):
        self.assertEqual(scan2.header_dict['S'],
                         scan2_1.header_dict['S'])
        self.assertEqual(scan2.header_dict['S'], "2 He")
        self.assertNotEqual(sf1["3.1"].header_dict['S'], 
                            scan2.header_dict['S'])
        self.assertEqual(scan2.header_dict['N'], '7')
        self.assertEqual(scan2.header_lines[1], '#N 7')
        self.assertEqual(len(scan2.header_lines), 3)   
        
    def test_file_headers(self):
        self.assertEqual(scan2.file_header_lines[0], 
                         '#F XCOM_CrossSections.dat')   
        self.assertEqual(len(scan2.file_header_lines), 6)
        # parsing headers with single character key 
        self.assertEqual(scan2.file_header_dict['F'], 
                         'XCOM_CrossSections.dat')   
        # parsing headers with long keys  
        self.assertEqual(scan2.file_header_dict['U03'], 
                         'XCOM itself can be found at:')
        # parsing empty headers
        self.assertEqual(scan2.file_header_dict['U02'], '')  
        
    def test_scan_labels(self):
        self.assertEqual(scan2.labels, 
                         ['PhotonEnergy[keV]', 'Rayleigh(coherent)[cm2/g]', 'Compton(incoherent)[cm2/g]', 'CoherentPlusIncoherent[cm2/g]', 'Photoelectric[cm2/g]', 'PairProduction[cm2/g]', 'TotalCrossSection[cm2/g]'])

    def test_data(self):
        # assertAlmostEqual compares 7 decimal places by default
        self.assertAlmostEqual(scan2.data_line(8)[2], 
                               0.11025)
        self.assertEqual(scan2.data.shape, (96, 7))
        self.assertEqual(scan2.nlines, 96)
        self.assertEqual(scan2.ncolumns, 7)
        self.assertAlmostEqual(numpy.sum(scan2.data), 
                               439208090.16178566)
        
    def test_motors(self):
        self.assertEqual(len(scan_w_motors.motor_names), 158)
        self.assertEqual(len(scan_w_motors.motor_positions), 158)
        self.assertAlmostEqual(sum(scan_w_motors.motor_positions), 
                               66338.5523591354)
        self.assertEqual(scan_w_motors.motor_names[1], 'Pslit Up')


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestSpecFile))
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestScan))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
