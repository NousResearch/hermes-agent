"""distutils.cygwinccompiler

Provides the CygwinCCompiler class, a subclass of UnixCCompiler that
handles the Cygwin port of the GNU C compiler to Windows.  It also contains
the Mingw32CCompiler class which handles the mingw32 port of GCC (same as
cygwin in no-cygwin mode).
"""

import copy
import os
import pathlib
import shlex
import sys
import warnings
from subprocess import check_output

from ...errors import (
    DistutilsExecError,
    DistutilsPlatformError,
)
from ...file_util import write_file
from ...sysconfig import get_config_vars
from ...version import LooseVersion, suppress_known_deprecation
from . import unix
from .errors import (
    CompileError,
    Error,
)


def get_msvcr():
    """No longer needed, but kept for backward compatibility."""
    return []


_runtime_library_dirs_msg = (
    "Unable to set runtime library search path on Windows, "
    "usually indicated by `runtime_library_dirs` parameter to Extension"
)


class Compiler(unix.Compiler):
    """Handles the Cygwin port of the GNU C compiler to Windows."""

    compiler_type = 'cygwin'
    obj_extension = ".o"
    static_lib_extension = ".a"
    shared_lib_extension = ".dll.a"
    dylib_lib_extension = ".dll"
    static_lib_format = "lib%s%s"
    shared_lib_format = "lib%s%s"
    dylib_lib_format = "cyg%s%s"
    exe_extension = ".exe"

    def __init__(self, verbose=False, dry_run=False, force=False):
        super().__init__(verbose, dry_run, force)

        status, details = check_config_h()
        self.debug_print(f"Python's GCC status: {status} (details: {details})")
        if status is not CONFIG_H_OK:
            self.warn(
                "Python's pyconfig.h doesn't seem to support your compiler. "
                f"Reason: {details}. "
                "Compiling may fail because of undefined preprocessor macros."
            )

        self.cc, self.cxx = get_config_vars('CC', 'CXX')

        # Override 'CC' and 'CXX' environment variables for
        # building using MINGW compiler for MSVC python.
        self.cc = os.environ.get('CC', self.cc or 'gcc')
        self.cxx = os.environ.get('CXX', self.cxx or 'g++')

        self.linker_dll = self.cc
        self.linker_dll_cxx = self.cxx
        shared_option = "-shared"

        self.set_executables(
            compiler=f'{self.cc} -mcygwin -O -Wall',
            compiler_so=f'{self.cc} -mcygwin -mdll -O -Wall',
            compiler_cxx=f'{self.cxx} -mcygwin -O -Wall',
            compiler_so_cxx=f'{self.cxx} -mcygwin -mdll -O -Wall',
            linker_exe=f'{self.cc} -mcygwin',
            linker_so=f'{self.linker_dll} -mcygwin {shared_option}',
            linker_exe_cxx=f'{self.cxx} -mcygwin',
            linker_so_cxx=f'{self.linker_dll_cxx} -mcygwin {shared_option}',
        )

        self.dll_libraries = get_msvcr()

    @property
    def gcc_version(self):
        # Older numpy depended on this existing to check for ancient
        # gcc versions. This doesn't make much sense with clang etc so
        # just hardcode to something recent.
        # https://github.com/numpy/numpy/pull/20333
        warnings.warn(
            "gcc_version attribute of CygwinCCompiler is deprecated. "
            "Instead of returning actual gcc version a fixed value 11.2.0 is returned.",
            DeprecationWarning,
            stacklevel=2,
        )
        with suppress_known_deprecation():
            return LooseVersion("11.2.0")

    def _compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
        """Compiles the source by spawning GCC and windres if needed."""
        if ext in ('.rc', '.res'):
            # gcc needs '.res' and '.rc' compiled to object files !!!
            try:
                self.spawn(["windres", "-i", src, "-o", obj])
            except DistutilsExecError as msg:
                raise CompileError(msg)
        else:  # for other files use the C-compiler
            try:
                if self.detect_language(src) == 'c++':
                    self.spawn(
                        self.compiler_so_cxx
                        + cc_args
                        + [src, '-o', obj]
                        + extra_postargs
                    )
                else:
                    self.spawn(
                        self.compiler_so + cc_args + [src, '-o', obj] + extra_postargs
                    )
            except DistutilsExecError as msg:
                raise CompileError(msg)

    def link(
        self,
        target_desc,
        objects,
        output_filename,
        output_dir=None,
        libraries=None,
        library_dirs=None,
        runtime_library_dirs=None,
        export_symbols=None,
        debug=False,
        extra_preargs=None,
        extra_postargs=None,
        build_temp=None,
        target_lang=None,
    ):
        """Link the objects."""
        # use separate copies, so we can modify the lists
        extra_preargs = copy.copy(extra_preargs or [])
        libraries = copy.copy(libraries or [])
        objects = copy.copy(objects or [])

        if runtime_library_dirs:
            self.warn(_runtime_library_dirs_msg)

        # Additional libraries
        libraries.extend(self.dll_libraries)

        # handle export symbols by creating a def-file
        # with executables this only works with gcc/ld as linker
        if (export_symbols is not None) and (
            target_desc != self.EXECUTABLE or self.linker_dll == "gcc"
        ):
            # (The linker doesn't do anything if output is up-to-date.
            # So it would probably better to check if we really need this,
            # but for this we had to insert some unchanged parts of
            # UnixCCompiler, and this is not what we want.)

            # we want to put some files in the same directory as the
            # object files are, build_temp doesn't help much
            # where are the object files
            temp_dir = os.path.dirname(objects[0])
            # name of dll to give the helper files the same base name
            (dll_name, dll_extension) = os.path.splitext(
                os.path.basename(output_filename)
            )

            # generate the filenames for these files
            def_file = os.path.join(temp_dir, dll_name + ".def")

            # Generate .def file
            contents = [f"LIBRARY {os.path.basename(output_filename)}", "EXPORTS"]
            contents.extend(export_symbols)
            self.execute(write_file, (def_file, contents), f"writing {def_file}")

            # next add options for def-file

            # for gcc/ld the def-file is specified as any object files
            objects.append(def_file)

        # end: if ((export_symbols is not None) and
        #        (target_desc != self.EXECUTABLE or self.linker_dll == "gcc")):

        # who wants symbols and a many times larger output file
        # should explicitly switch the debug mode on
        # otherwise we let ld strip the output file
        # (On my machine: 10KiB < stripped_file < ??100KiB
        #   unstripped_file = stripped_file + XXX KiB
        #  ( XXX=254 for a typical python extension))
        if not debug:
            extra_preargs.append("-s")

        super().link(
            target_desc,
            objects,
            output_filename,
            output_dir,
            libraries,
            library_dirs,
            runtime_library_dirs,
            None,  # export_symbols, we do this in our def-file
            debug,
            extra_preargs,
            extra_postargs,
            build_temp,
            target_lang,
        )

    def runtime_library_dir_option(self, dir):
        # cygwin doesn't support rpath. While in theory we could error
        # out like MSVC does, code might expect it to work like on Unix, so
        # just warn and hope for the best.
        self.warn(_runtime_library_dirs_msg)
        return []

    # -- Miscellaneous methods -----------------------------------------

    def _make_out_path(self, output_dir, strip_dir, src_name):
        # use normcase to make sure '.rc' is really '.rc' and not '.RC'
        norm_src_name = os.path.normcase(src_name)
        return super()._make_out_path(output_dir, strip_dir, norm_src_name)

    @property
    def out_extensions(self):
        """
        Add support for rc and res files.
        """
        return {
            **super().out_extensions,
            **{ext: ext + self.obj_extension for ext in ('.res', '.rc')},
        }


# the same as cygwin plus some additional parameters
class MinGW32Compiler(Compiler):
    """Handles the Mingw32 port of the GNU C compiler to Windows."""

    compiler_type = 'mingw32'

    def __init__(self, verbose=False, dry_run=False, force=False):
        super().__init__(verbose, dry_run, force)

        shared_option = "-shared"

        if is_cygwincc(self.cc):
            raise Error('Cygwin gcc cannot be used with --compiler=mingw32')

        self.set_executables(
            compiler=f'{self.cc} -O -Wall',
            compiler_so=f'{self.cc} -shared -O -Wall',
            compiler_so_cxx=f'{self.cxx} -shared -O -Wall',
            compiler_cxx=f'{self.cxx} -O -Wall',
            linker_exe=f'{self.cc}',
            linker_so=f'{self.linker_dll} {shared_option}',
            linker_exe_cxx=f'{self.cxx}',
            linker_so_cxx=f'{self.linker_dll_cxx} {shared_option}',
        )

    def runtime_library_dir_option(self, dir):
        raise DistutilsPlatformError(_runtime_library_dirs_msg)


# Because these compilers aren't configured in Python's pyconfig.h file by
# default, we should at least warn the user if he is using an unmodified
# version.

CONFIG_H_OK = "ok"
CONFIG_H_NOTOK = "not ok"
CONFIG_H_UNCERTAIN = "uncertain"


def check_config_h():
    """Check if the current Python installation appears amenable to building
    extensions with GCC.

    Returns a tuple (status, details), where 'status' is one of the following
    constants:

    - CONFIG_H_OK: all is well, go ahead and compile
    - CONFIG_H_NOTOK: doesn't look good
    - CONFIG_H_UNCERTAIN: not sure -- unable to read pyconfig.h

    'details' is a human-readable string explaining the situation.

    Note there are two ways to conclude "OK": either 'sys.version' contains
    the string "GCC" (implying that this Python was built with GCC), or the
    installed "pyconfig.h" contains the string "__GNUC__".
    """

    # XXX since this function also checks sys.version, it's not strictly a
    # "pyconfig.h" check -- should probably be renamed...

    from distutils import sysconfig

    # if sys.version contains GCC then python was compiled with GCC, and the
    # pyconfig.h file should be OK
    if "GCC" in sys.version:
        return CONFIG_H_OK, "sys.version mentions 'GCC'"

    # Clang would also work
    if "Clang" in sys.version:
        return CONFIG_H_OK, "sys.version mentions 'Clang'"

    # let's see if __GNUC__ is mentioned in python.h
    fn = sysconfig.get_config_h_filename()
    try:
        config_h = pathlib.Path(fn).read_text(encoding='utf-8')
    except OSError as exc:
        return (CONFIG_H_UNCERTAIN, f"couldn't read '{fn}': {exc.strerror}")
    else:
        substring = '__GNUC__'
        if substring in config_h:
            code = CONFIG_H_OK
            mention_inflected = 'mentions'
        else:
            code = CONFIG_H_NOTOK
            mention_inflected = 'does not mention'
        return code, f"{fn!r} {mention_inflected} {substring!r}"


def is_cygwincc(cc):
    """Try to determine if the compiler that would be used is from cygwin."""
    out_string = check_output(shlex.split(cc) + ['-dumpmachine'])
    return out_string.strip().endswith(b'cygwin')


get_versions = None
"""
A stand-in for the previous get_versions() function to prevent failures
when monkeypatched. See pypa/setuptools#2969.
"""
