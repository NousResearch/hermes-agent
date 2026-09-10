"""distutils.zosccompiler

Contains the selection of the c & c++ compilers on z/OS. There are several
different c compilers on z/OS, all of them are optional, so the correct
one needs to be chosen based on the users input. This is compatible with
the following compilers:

IBM C/C++ For Open Enterprise Languages on z/OS 2.0
IBM Open XL C/C++ 1.1 for z/OS
IBM XL C/C++ V2.4.1 for z/OS 2.4 and 2.5
IBM z/OS XL C/C++
"""

import os

from ... import sysconfig
from ...errors import DistutilsExecError
from . import unix
from .errors import CompileError

_cc_args = {
    'ibm-openxl': [
        '-m64',
        '-fvisibility=default',
        '-fzos-le-char-mode=ascii',
        '-fno-short-enums',
    ],
    'ibm-xlclang': [
        '-q64',
        '-qexportall',
        '-qascii',
        '-qstrict',
        '-qnocsect',
        '-Wa,asa,goff',
        '-Wa,xplink',
        '-qgonumber',
        '-qenum=int',
        '-Wc,DLL',
    ],
    'ibm-xlc': [
        '-q64',
        '-qexportall',
        '-qascii',
        '-qstrict',
        '-qnocsect',
        '-Wa,asa,goff',
        '-Wa,xplink',
        '-qgonumber',
        '-qenum=int',
        '-Wc,DLL',
        '-qlanglvl=extc99',
    ],
}

_cxx_args = {
    'ibm-openxl': [
        '-m64',
        '-fvisibility=default',
        '-fzos-le-char-mode=ascii',
        '-fno-short-enums',
    ],
    'ibm-xlclang': [
        '-q64',
        '-qexportall',
        '-qascii',
        '-qstrict',
        '-qnocsect',
        '-Wa,asa,goff',
        '-Wa,xplink',
        '-qgonumber',
        '-qenum=int',
        '-Wc,DLL',
    ],
    'ibm-xlc': [
        '-q64',
        '-qexportall',
        '-qascii',
        '-qstrict',
        '-qnocsect',
        '-Wa,asa,goff',
        '-Wa,xplink',
        '-qgonumber',
        '-qenum=int',
        '-Wc,DLL',
        '-qlanglvl=extended0x',
    ],
}

_asm_args = {
    'ibm-openxl': ['-fasm', '-fno-integrated-as', '-Wa,--ASA', '-Wa,--GOFF'],
    'ibm-xlclang': [],
    'ibm-xlc': [],
}

_ld_args = {
    'ibm-openxl': [],
    'ibm-xlclang': ['-Wl,dll', '-q64'],
    'ibm-xlc': ['-Wl,dll', '-q64'],
}


# Python on z/OS is built with no compiler specific options in it's CFLAGS.
# But each compiler requires it's own specific options to build successfully,
# though some of the options are common between them
class Compiler(unix.Compiler):
    src_extensions = ['.c', '.C', '.cc', '.cxx', '.cpp', '.m', '.s']
    _cpp_extensions = ['.cc', '.cpp', '.cxx', '.C']
    _asm_extensions = ['.s']

    def _get_zos_compiler_name(self):
        zos_compiler_names = [
            os.path.basename(binary)
            for envvar in ('CC', 'CXX', 'LDSHARED')
            if (binary := os.environ.get(envvar, None))
        ]
        if len(zos_compiler_names) == 0:
            return 'ibm-openxl'

        zos_compilers = {}
        for compiler in (
            'ibm-clang',
            'ibm-clang64',
            'ibm-clang++',
            'ibm-clang++64',
            'clang',
            'clang++',
            'clang-14',
        ):
            zos_compilers[compiler] = 'ibm-openxl'

        for compiler in ('xlclang', 'xlclang++', 'njsc', 'njsc++'):
            zos_compilers[compiler] = 'ibm-xlclang'

        for compiler in ('xlc', 'xlC', 'xlc++'):
            zos_compilers[compiler] = 'ibm-xlc'

        return zos_compilers.get(zos_compiler_names[0], 'ibm-openxl')

    def __init__(self, verbose=False, dry_run=False, force=False):
        super().__init__(verbose, dry_run, force)
        self.zos_compiler = self._get_zos_compiler_name()
        sysconfig.customize_compiler(self)

    def _compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
        local_args = []
        if ext in self._cpp_extensions:
            compiler = self.compiler_cxx
            local_args.extend(_cxx_args[self.zos_compiler])
        elif ext in self._asm_extensions:
            compiler = self.compiler_so
            local_args.extend(_cc_args[self.zos_compiler])
            local_args.extend(_asm_args[self.zos_compiler])
        else:
            compiler = self.compiler_so
            local_args.extend(_cc_args[self.zos_compiler])
        local_args.extend(cc_args)

        try:
            self.spawn(compiler + local_args + [src, '-o', obj] + extra_postargs)
        except DistutilsExecError as msg:
            raise CompileError(msg)

    def runtime_library_dir_option(self, dir):
        return '-L' + dir

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
        # For a built module to use functions from cpython, it needs to use Pythons
        # side deck file. The side deck is located beside the libpython3.xx.so
        ldversion = sysconfig.get_config_var('LDVERSION')
        if sysconfig.python_build:
            side_deck_path = os.path.join(
                sysconfig.get_config_var('abs_builddir'),
                f'libpython{ldversion}.x',
            )
        else:
            side_deck_path = os.path.join(
                sysconfig.get_config_var('installed_base'),
                sysconfig.get_config_var('platlibdir'),
                f'libpython{ldversion}.x',
            )

        if os.path.exists(side_deck_path):
            if extra_postargs:
                extra_postargs.append(side_deck_path)
            else:
                extra_postargs = [side_deck_path]

        # Check and replace libraries included side deck files
        if runtime_library_dirs:
            for dir in runtime_library_dirs:
                for library in libraries[:]:
                    library_side_deck = os.path.join(dir, f'{library}.x')
                    if os.path.exists(library_side_deck):
                        libraries.remove(library)
                        extra_postargs.append(library_side_deck)
                        break

        # Any required ld args for the given compiler
        extra_postargs.extend(_ld_args[self.zos_compiler])

        super().link(
            target_desc,
            objects,
            output_filename,
            output_dir,
            libraries,
            library_dirs,
            runtime_library_dirs,
            export_symbols,
            debug,
            extra_preargs,
            extra_postargs,
            build_temp,
            target_lang,
        )
