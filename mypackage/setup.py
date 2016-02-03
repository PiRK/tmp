#!/usr/bin/python
import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

my_dir = os.path.dirname(os.path.abspath(__file__))   # setup.py dir
include_dirs = [os.path.join(my_dir, "cstuff")]       # .h dir
sources = [os.path.join(my_dir, "cstuff", "mymax.c"), # .c 
           os.path.join(my_dir, "mymax_.pyx"),        # .pyx  
           ]

cy_mod=Extension("mymax",                     # module name (from mymax import cmax)
              sources=sources,
              language="c",
              include_dirs=include_dirs,
              )

ext_modules = [cy_mod]

setup(ext_modules=ext_modules, 
      cmdclass={'build_ext': build_ext}
      )
