#pxd
cimport cython

cdef extern from "SpecFile.h":
    struct _SpecFile:
        pass
ctypedef _SpecFile SpecFile

cdef extern from "SpecFile.h":
    SpecFile* SfOpen(char*, int*)
    long SfScanNo(SpecFile*)
    int SfClose(SpecFile*)
    int SfData(SpecFile*, long, double***, long**, int*)
    long *SfList(SpecFile*, int*)
    

