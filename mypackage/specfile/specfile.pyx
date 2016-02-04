#pyx
from aifc import data
cimport cython
from specfile_pxd cimport *
cimport numpy
import numpy

#TODO: SfList
#SfData



cdef class PySpecFile(object):
    cdef SpecFile *_sf
    cdef int _error
    
    def __init__(self, filename):
        self._sf =  SfOpen(filename, &self._error)

    def __len__(self):
        '''returns number of scans'''
        return SfScanNo(self._sf)

    def __dealloc__(self):
        #TODO check what to do with returned value
        print(" Passing by")
        if SfClose(self._sf):
            print(" ERROR cleaning up")
            
    def data(self, scan_no):
        cdef: 
            #double[:,:,:] mydata 
            double*** mydata
            #numpy.ndarray[dtype=numpy.float64_t, ndim=3] mydata 
            long** data_info
            
        sfdata_error = SfData(self._sf, 
                              scan_no, 
                              mydata, 
                              data_info, 
                              &self._error)
         
        if sfdata_error:
            print "error"
            
        return mydata 
    #how can i read this double*** in python?????
    # using views?
    # retrieve no_lines and no_columns before using SfData?
        
#from cpython cimport array
#import array
#cdef array.array a = array.array('d', [1, 2, 3])