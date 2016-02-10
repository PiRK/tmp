#/*##########################################################################
# Copyright (C) 2004-2016 European Synchrotron Radiation Facility, Grenoble, France
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
#############################################################################*/
from PyMca5.PyMcaIO.specfilewrapper import Specfile
__author__ = "P. Knobel - ESRF Data Analysis"
__contact__ = "pierre.knobel@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__doc__ = """
This module is a cython binding to wrap the C-library SpecFile.

Classes
=======

- :class:`PySpecFile`
"""
import os.path
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

debugging = True

def debug_msg(msg):
    if debugging:
        print("Debug message: " + str(msg))

SF_ERR_NO_ERRORS = 0
SF_ERR_MEMORY_ALLOC = 1
SF_ERR_FILE_OPEN = 2
SF_ERR_FILE_CLOSE = 3
SF_ERR_FILE_READ = 4
SF_ERR_FILE_WRITE = 5
SF_ERR_LINE_NOT_FOUND = 6
SF_ERR_SCAN_NOT_FOUND = 7
SF_ERR_HEADER_NOT_FOUND = 8
SF_ERR_LABEL_NOT_FOUND = 9
SF_ERR_MOTOR_NOT_FOUND = 10
SF_ERR_POSITION_NOT_FOUND = 11
SF_ERR_LINE_EMPTY = 12
SF_ERR_USER_NOT_FOUND = 13
SF_ERR_COL_NOT_FOUND = 14
SF_ERR_MCA_NOT_FOUND = 15

cdef class Scan(object):
    def __init__(self, specfile):
        self._specfile = specfile
        


cdef class SpecFile(object):
    '''
    Accessing SpecFiles

    :param filename: Path of the SpecFile to read
    :type label_text: string
    '''
    
    cdef SpecFileHandle *handle   #SpecFile struct in SpecFile.h
    cdef int _error
   
    def __cinit__(self, filename):
        if os.path.isfile(filename):
            self.handle =  SfOpen(filename, &self._error)
        else:
            self._error = SF_ERR_FILE_OPEN
       
    def __init__(self, filename):            
        if self._error:
            raise IOError(self.get_error_string())
        
    def __len__(self):
        '''returns the number of scans in the SpecFile'''
        return SfScanNo(self.handle)

    def __dealloc__(self):
        '''Destructor: Calls SfClose(self.handle)'''
        debug_msg(" Passing by destructor")
        if not self._error == SF_ERR_FILE_OPEN:
            if SfClose(self.handle):
                self._error = SF_ERR_FILE_CLOSE
                print("Error while closing")
        
    def get_error_string(self):
        '''Returns the error message corresponding to the error code
        contained in self._error.
        
        :param code: Error code 
        :type code: int
        '''    
        return (<bytes> SfError(self._error)).encode('utf-8)') 
    
    def index(self, scan_number, scan_order=1):
        '''Returns scan index from scan number and order.
        
        :param scan_number: Scan number (possibly non-unique). 
        :type scan_number: int
        :param scan_order: Scan order. 
        :type scan_order: int default 1
        :returns: Unique scan index
        :rtype: int
        
        
        Scan indices are increasing from 1 to len(self) in the order in which
        they appear in the file.
        Scan numbers are defined by users and are not necessarily unique.
        The scan order for a given scan number increments each time the scan 
        number appers in a given file.'''
        idx = SfIndex(self.handle, scan_number, scan_order)
        if idx == -1:
            self._error = SF_ERR_SCAN_NOT_FOUND
            raise IOError(self.get_error_string())
        return idx
    
    def number(self, scan_index):
        '''Returns scan number from scan index.
        
        :param scan_index: Unique scan index between 1 and len(self). 
        :type scan_index: int
        :returns: User defined scan number.
        :rtype: int
        '''
        idx = SfNumber(self.handle, scan_index)
        if idx == -1:
            self._error = SF_ERR_SCAN_NOT_FOUND
            raise IOError(self.get_error_string())
        return idx
    
    def order(self, scan_index):
        '''Returns scan order from scan index.
        
        :param scan_index: Unique scan index between 1 and len(self). 
        :type scan_index: int
        :returns: Scan order (sequential number incrementing each time a 
                 non-unique occurrence of a scan number is encountered).
        :rtype: int
        '''
        ordr = SfOrder(self.handle, scan_index)
        if ordr == -1:
            self._error = SF_ERR_SCAN_NOT_FOUND
            raise IOError(self.get_error_string())
        return ordr
            
    def data(self, scan_no): # TODO: move to Scan class
        '''Returns data and metadata for the specified scan number.
        
        :param scan_no: Scan number to return. 
        :type scan_no: int
        :return: ret_array
        :rtype: numpy.ndarray 
        
        Example:
        --------
        
        .. code-block:: python
            
            from specfile import PySpecFile
            sf = PySpecFile("t.dat")
            sfdata = sf.data(2)
            nlines, ncolumns = sfdata.shape
        '''        
        cdef: 
            double** mydata
            long* data_info
            int i, j
            long nlines, ncolumns, regular
            
        sfdata_error = SfData(self.handle, 
                              scan_no, 
                              &mydata, 
                              &data_info, 
                              &self._error)
                  
        if sfdata_error:
            raise IOError(self.get_error_string())
        
        nlines = data_info[0] 
        ncolumns = data_info[1]
        regular = data_info[2]
        
        cdef numpy.ndarray ret_array = numpy.empty((nlines, ncolumns), 
                                                   dtype=numpy.double)
        for i in range(nlines):
            for j in range(ncolumns):
                ret_array[i, j] = mydata[i][j]    
        
        free(mydata)
        free(data_info)
        
        # nlines and ncolumns can be accessed as ret_array.shape
        return ret_array

    def list(self): 
        '''Returns list (1D numpy array) of scan indexes in SpecFile.
                
        :param scan_no: Scan number to return. 
        :type scan_no: int
        :return: retArray
        :rtype: numpy array 
        '''    
        cdef long *indexes
        indexes = SfList(self.handle, &self._error)
        
        if self._error:
            raise IOError(self.get_error_string())
        
        n_indexes = len(self)
        retArray = numpy.empty((n_indexes,),dtype=numpy.int)
        for i in range(n_indexes):
            retArray[i] = indexes[i]
        free(indexes)
        
        return retArray
    
    def columns(self, scan_no): # TODO: move to Scan class
        '''Return number of columns in a scan from the #N header line
        (without #N and ssan number)
        
        :param scan_no: Scan number 
        :type scan_no: int
        :return: Number of columns in scan from #N record
        :rtype: int
        '''
        no_columns = SfNoColumns(self.handle, scan_no, &self._error)
        if self._error:
            raise IOError(self.get_error_string())
        
        return no_columns
        
    def command(self, scan_no): # TODO: move to Scan class
        '''Return #S line (without #S and scan number)
        
        :param scan_no: Scan number
        :type scan_no: int
        :return: S line
        :rtype: utf-8 encoded bytes
        '''
        s_record = <bytes> SfCommand(self.handle, scan_no, &self._error)
        if self._error:
            raise IOError(self.get_error_string())
        
        return s_record.encode('utf-8)')
    
    def date(self, scan_no):   #TODO: segmentation fault when #D line absent
         # TODO: move to Scan class
        '''Return date from #D line
        
        :param scan_no: Scan number
        :type scan_no: int
        :return: Date from #D line
        :rtype: utf-8 encoded bytes
        '''
        d_record = <bytes> SfDate(self.handle, scan_no, &self._error)
        if self._error:
            raise IOError(self.get_error_string())
        
        return d_record.encode('utf-8')
    
    def __getitem__(self, key):   #TODO: everything
        '''Return a Scan object
        
        Example:
        --------
        
        .. code-block:: python
            
            from specfile import SpecFile
            sf = SpecFile("t.dat")
            myscan = sf[2]
            nlines, ncolumns = myscan.data.shape
        '''
        #if isinstance(key, int):
        #    # check in range
        #return Scan(self)
        pass
    
    
    
    
    
