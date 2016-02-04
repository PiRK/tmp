#pyx
cimport cython
from specfile cimport *

def specfile_open(filename):
    error = 0
    sf = SfOpen(filename, error))
    sf.length = SfScanNo(sf)
    sf.name = filename
    
    return sf