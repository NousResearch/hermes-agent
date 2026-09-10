# required for older numpy versions on Pythons prior to 3.12; see pypa/setuptools#4876
from ..compilers.C.base import _default_compilers, compiler_class  # noqa: F401
