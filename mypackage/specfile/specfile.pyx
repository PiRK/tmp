
cimport cython

from libc.stdlib cimport free, malloc
from libc.string cimport memcpy

cimport numpy
import numpy
numpy.import_array()

from specfile_pxd cimport *

#TODO: SfList
#SfData

# cdef class ArrayWrapper:
#     # Author: Gael Varoquaux
#     # License: BSD
#     # url: http://gael-varoquaux.info/programming/cython-example-of-exposing-c-computed-arrays-in-python-without-data-copies.html?p=157
#     cdef void* data_ptr
#     cdef int size
# 
#     cdef set_data(self, int size, void* data_ptr):
#         """ Set the data of the array
#         This cannot be done in the constructor as it must recieve C-level
#         arguments.
#         Parameters:
#         -----------
#         size: int
#             Length of the array.
#         data_ptr: void*
#             Pointer to the data            
#         """
#         self.data_ptr = data_ptr
#         self.size = size
# 
#     def __array__(self):
#         """ Here we use the __array__ method, that is called when numpy
#             tries to get an array from the object."""
#         cdef numpy.npy_intp shape[1]
#         shape[0] = <numpy.npy_intp> self.size
#         # Create a 1D array, of length 'size'
#         ndarray = numpy.PyArray_SimpleNewFromData(1, shape,
#                                                   numpy.NPY_INT, self.data_ptr)
#         return ndarray
# 
#     def __dealloc__(self):
#         """ Frees the array. This is called by Python when all the
#         references to the object are gone. """
#         free(<void*>self.data_ptr)


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
            double** mydata
            long* data_info
            long long_scan_no = scan_no
            #long nlines, ncolumns, regular
            
        sfdata_error = SfData(self._sf, 
                              long_scan_no, 
                              &mydata, 
                              &data_info, 
                              &self._error)
          
        if sfdata_error:
            print("error ", str(sfdata_error)) #TODO: handle error
        
        nlines = data_info[0] 
        ncolumns = data_info[1]
        regular = data_info[2]
        
        ret_array = numpy.empty((nlines, ncolumns), dtype=numpy.double)
        for i in range(nlines):
            for j in range(ncolumns):
                ret_array[i, j] = mydata[i][j]
        
        
        # Alternative 2  (obscure compilation  error message  compilation: expected identifier or ‘(’ before numeric constant)
        #cdef double **ret_array = <double **>malloc(nlines * sizeof(double *))
        #for i in range(ncolumns):
        #    ret_array[i] = <double *>malloc(ncolumns * sizeof(double))
        #ret_array = numpy.empty((nlines, ncolumns), dtype=numpy.double)
        #cdef double[:, :] c_array = ret_array
        #memcpy(ret_array, 
        #       mydata, 
        #       nlines * ncolumns * sizeof(double))
        
        # Alternative 3: error message Pointer base type does not match cython.array base type
        #ret_array = numpy.asarray(<numpy.double_t[:nlines, :ncolumns]> mydata)

        free(mydata)
        free(data_info)
        
        return (nlines, ncolumns, regular), ret_array

    def list(self):
        cdef long *indexes
        indexes = SfList(self._sf, &self._error)
        n_indexes = len(self)
        retArray = numpy.empty((n_indexes,),dtype=numpy.int)
        for i in range(n_indexes):
            retArray[i] = indexes[i]
        free(indexes)
        return retArray
    
   