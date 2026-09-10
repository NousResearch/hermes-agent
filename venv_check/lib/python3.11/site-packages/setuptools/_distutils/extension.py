"""distutils.extension

Provides the Extension class, used to describe C/C++ extension
modules in setup scripts."""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterable

# This class is really only used by the "build_ext" command, so it might
# make sense to put it in distutils.command.build_ext.  However, that
# module is already big enough, and I want to make this class a bit more
# complex to simplify some common cases ("foo" module in "foo.c") and do
# better error-checking ("foo.c" actually exists).
#
# Also, putting this in build_ext.py means every setup script would have to
# import that large-ish module (indirectly, through distutils.core) in
# order to do anything.


class Extension:
    """Just a collection of attributes that describes an extension
    module and everything needed to build it (hopefully in a portable
    way, but there are hooks that let you be as unportable as you need).

    Instance attributes:
      name : string
        the full name of the extension, including any packages -- ie.
        *not* a filename or pathname, but Python dotted name
      sources : Iterable[string | os.PathLike]
        iterable of source filenames (except strings, which could be misinterpreted
        as a single filename), relative to the distribution root (where the setup
        script lives), in Unix form (slash-separated) for portability. Can be any
        non-string iterable (list, tuple, set, etc.) containing strings or
        PathLike objects. Source files may be C, C++, SWIG (.i), platform-specific
        resource files, or whatever else is recognized by the "build_ext" command
        as source for a Python extension.
      include_dirs : [string]
        list of directories to search for C/C++ header files (in Unix
        form for portability)
      define_macros : [(name : string, value : string|None)]
        list of macros to define; each macro is defined using a 2-tuple,
        where 'value' is either the string to define it to or None to
        define it without a particular value (equivalent of "#define
        FOO" in source or -DFOO on Unix C compiler command line)
      undef_macros : [string]
        list of macros to undefine explicitly
      library_dirs : [string]
        list of directories to search for C/C++ libraries at link time
      libraries : [string]
        list of library names (not filenames or paths) to link against
      runtime_library_dirs : [string]
        list of directories to search for C/C++ libraries at run time
        (for shared extensions, this is when the extension is loaded)
      extra_objects : [string]
        list of extra files to link with (eg. object files not implied
        by 'sources', static library that must be explicitly specified,
        binary resource files, etc.)
      extra_compile_args : [string]
        any extra platform- and compiler-specific information to use
        when compiling the source files in 'sources'.  For platforms and
        compilers where "command line" makes sense, this is typically a
        list of command-line arguments, but for other platforms it could
        be anything.
      extra_link_args : [string]
        any extra platform- and compiler-specific information to use
        when linking object files together to create the extension (or
        to create a new static Python interpreter).  Similar
        interpretation as for 'extra_compile_args'.
      export_symbols : [string]
        list of symbols to be exported from a shared extension.  Not
        used on all platforms, and not generally necessary for Python
        extensions, which typically export exactly one symbol: "init" +
        extension_name.
      swig_opts : [string]
        any extra options to pass to SWIG if a source file has the .i
        extension.
      depends : [string]
        list of files that the extension depends on
      language : string
        extension language (i.e. "c", "c++", "objc"). Will be detected
        from the source extensions if not provided.
      optional : boolean
        specifies that a build failure in the extension should not abort the
        build process, but simply not install the failing extension.
    """

    # When adding arguments to this constructor, be sure to update
    # setup_keywords in core.py.
    def __init__(
        self,
        name: str,
        sources: Iterable[str | os.PathLike[str]],
        include_dirs: list[str] | None = None,
        define_macros: list[tuple[str, str | None]] | None = None,
        undef_macros: list[str] | None = None,
        library_dirs: list[str] | None = None,
        libraries: list[str] | None = None,
        runtime_library_dirs: list[str] | None = None,
        extra_objects: list[str] | None = None,
        extra_compile_args: list[str] | None = None,
        extra_link_args: list[str] | None = None,
        export_symbols: list[str] | None = None,
        swig_opts: list[str] | None = None,
        depends: list[str] | None = None,
        language: str | None = None,
        optional: bool | None = None,
        **kw,  # To catch unknown keywords
    ):
        if not isinstance(name, str):
            raise TypeError("'name' must be a string")

        # handle the string case first; since strings are iterable, disallow them
        if isinstance(sources, str):
            raise TypeError(
                "'sources' must be an iterable of strings or PathLike objects, not a string"
            )

        # now we check if it's iterable and contains valid types
        try:
            self.sources = list(map(os.fspath, sources))
        except TypeError:
            raise TypeError(
                "'sources' must be an iterable of strings or PathLike objects"
            )

        self.name = name
        self.include_dirs = include_dirs or []
        self.define_macros = define_macros or []
        self.undef_macros = undef_macros or []
        self.library_dirs = library_dirs or []
        self.libraries = libraries or []
        self.runtime_library_dirs = runtime_library_dirs or []
        self.extra_objects = extra_objects or []
        self.extra_compile_args = extra_compile_args or []
        self.extra_link_args = extra_link_args or []
        self.export_symbols = export_symbols or []
        self.swig_opts = swig_opts or []
        self.depends = depends or []
        self.language = language
        self.optional = optional

        # If there are unknown keyword options, warn about them
        if len(kw) > 0:
            options = [repr(option) for option in kw]
            options = ', '.join(sorted(options))
            msg = f"Unknown Extension options: {options}"
            warnings.warn(msg)

    def __repr__(self):
        return f'<{self.__class__.__module__}.{self.__class__.__qualname__}({self.name!r}) at {id(self):#x}>'


def read_setup_file(filename):  # noqa: C901
    """Reads a Setup file and returns Extension instances."""
    from distutils.sysconfig import _variable_rx, expand_makefile_vars, parse_makefile
    from distutils.text_file import TextFile
    from distutils.util import split_quoted

    # First pass over the file to gather "VAR = VALUE" assignments.
    vars = parse_makefile(filename)

    # Second pass to gobble up the real content: lines of the form
    #   <module> ... [<sourcefile> ...] [<cpparg> ...] [<library> ...]
    file = TextFile(
        filename,
        strip_comments=True,
        skip_blanks=True,
        join_lines=True,
        lstrip_ws=True,
        rstrip_ws=True,
    )
    try:
        extensions = []

        while True:
            line = file.readline()
            if line is None:  # eof
                break
            if _variable_rx.match(line):  # VAR=VALUE, handled in first pass
                continue

            if line[0] == line[-1] == "*":
                file.warn(f"'{line}' lines not handled yet")
                continue

            line = expand_makefile_vars(line, vars)
            words = split_quoted(line)

            # NB. this parses a slightly different syntax than the old
            # makesetup script: here, there must be exactly one extension per
            # line, and it must be the first word of the line.  I have no idea
            # why the old syntax supported multiple extensions per line, as
            # they all wind up being the same.

            module = words[0]
            ext = Extension(module, [])
            append_next_word = None

            for word in words[1:]:
                if append_next_word is not None:
                    append_next_word.append(word)
                    append_next_word = None
                    continue

                suffix = os.path.splitext(word)[1]
                switch = word[0:2]
                value = word[2:]

                if suffix in (".c", ".cc", ".cpp", ".cxx", ".c++", ".m", ".mm"):
                    # hmm, should we do something about C vs. C++ sources?
                    # or leave it up to the CCompiler implementation to
                    # worry about?
                    ext.sources.append(word)
                elif switch == "-I":
                    ext.include_dirs.append(value)
                elif switch == "-D":
                    equals = value.find("=")
                    if equals == -1:  # bare "-DFOO" -- no value
                        ext.define_macros.append((value, None))
                    else:  # "-DFOO=blah"
                        ext.define_macros.append((value[0:equals], value[equals + 2 :]))
                elif switch == "-U":
                    ext.undef_macros.append(value)
                elif switch == "-C":  # only here 'cause makesetup has it!
                    ext.extra_compile_args.append(word)
                elif switch == "-l":
                    ext.libraries.append(value)
                elif switch == "-L":
                    ext.library_dirs.append(value)
                elif switch == "-R":
                    ext.runtime_library_dirs.append(value)
                elif word == "-rpath":
                    append_next_word = ext.runtime_library_dirs
                elif word == "-Xlinker":
                    append_next_word = ext.extra_link_args
                elif word == "-Xcompiler":
                    append_next_word = ext.extra_compile_args
                elif switch == "-u":
                    ext.extra_link_args.append(word)
                    if not value:
                        append_next_word = ext.extra_link_args
                elif suffix in (".a", ".so", ".sl", ".o", ".dylib"):
                    # NB. a really faithful emulation of makesetup would
                    # append a .o file to extra_objects only if it
                    # had a slash in it; otherwise, it would s/.o/.c/
                    # and append it to sources.  Hmmmm.
                    ext.extra_objects.append(word)
                else:
                    file.warn(f"unrecognized argument '{word}'")

            extensions.append(ext)
    finally:
        file.close()

    return extensions
