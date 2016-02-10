#pxd
cimport cython

cdef extern from "SpecFile.h":
    struct _SpecFile:
        pass
ctypedef _SpecFile SpecFile

cdef extern from "SpecFile.h":
    # sfinit
    SpecFile* SfOpen(char*, int*)
    int SfClose(SpecFile*)
    char *SfError(int)
    
    # sfindex
    long *SfList(SpecFile*, int*)
    long SfScanNo(SpecFile*)
    
    # sfdata
    int SfData(SpecFile*, long, double***, long**, int*)
    
    # sfheader
    char *SfCommand(SpecFile*, long, int*)
    long SfNoColumns(SpecFile*, long, int*)
    char *SfDate(SpecFile*, long, int*)
    

