import importlib

from .compilers.C import unix

UnixCCompiler = unix.Compiler

# ensure import of unixccompiler implies ccompiler imported
# (pypa/setuptools#4871)
importlib.import_module('distutils.ccompiler')
