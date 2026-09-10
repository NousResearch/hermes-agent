import warnings

from .compilers.C import msvc

__all__ = ["MSVCCompiler"]

MSVCCompiler = msvc.Compiler


def __getattr__(name):
    if name == '_get_vc_env':
        warnings.warn(
            "_get_vc_env is private; find an alternative (pypa/distutils#340)"
        )
        return msvc._get_vc_env
    raise AttributeError(name)
