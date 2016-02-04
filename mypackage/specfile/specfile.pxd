#pxd
cimport cython

cdef extern from "SpecFile.h":
    SpecFile* SfOpen(char*, int*)
    long SfScanNo(SpecFile*)

ctypedef SpecFile:
    pass