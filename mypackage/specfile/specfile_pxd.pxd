#pxd
cimport cython

cdef extern from "SpecFile.h":
    struct _SpecFile:
        pass
ctypedef _SpecFile SpecFileHandle

cdef extern from "SpecFile.h":
    # sfinit
    SpecFileHandle* SfOpen(char*, int*)
    int SfClose(SpecFileHandle*)
    char *SfError(int)
    
    # sfindex
    long *SfList(SpecFileHandle*, int*)
    long SfScanNo(SpecFileHandle*)
    long SfIndex(SpecFileHandle*, long, long)
    long SfNumber(SpecFileHandle*, long)
    long SfOrder(SpecFileHandle*, long)
    
    # sfdata
    int SfData(SpecFileHandle*, long, double***, long**, int*)
    
    # sfheader
    char *SfCommand(SpecFileHandle*, long, int*)
    long SfNoColumns(SpecFileHandle*, long, int*)
    char *SfDate(SpecFileHandle*, long, int*)
    

