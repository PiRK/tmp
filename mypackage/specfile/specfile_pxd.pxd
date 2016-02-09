#pxd
cimport cython

cdef extern from "SpecFile.h":
    struct _SpecFile:
        pass
ctypedef _SpecFile SpecFile

cdef extern from "SpecFile.h":
    SpecFile* SfOpen(char*, int*)
    int SfClose(SpecFile*)
    int SfData(SpecFile*, long, double***, long**, int*)
    long *SfList(SpecFile*, int*)
    char *SfError(int)
    long SfScanNo(SpecFile*)
    

