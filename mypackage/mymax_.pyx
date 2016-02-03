cimport cython
from mymax cimport *  # (imports from function exposed in mymax.pxd)


def pymax(float a, float b):
    print("print from python function wrapping a c function")
    return cmax(a, b)
    