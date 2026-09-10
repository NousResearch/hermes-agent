"""distutils.command.install

Implements the Distutils 'install' command."""

from __future__ import annotations

import collections
import contextlib
import itertools
import os
import sys
import sysconfig
from distutils._log import log
from site import USER_BASE, USER_SITE
from typing import ClassVar

from ..core import Command
from ..debug import DEBUG
from ..errors import DistutilsOptionError, DistutilsPlatformError
from ..file_util import write_file
from ..sysconfig import get_config_vars
from ..util import change_root, convert_path, get_platform, subst_vars
from . import _framework_compat as fw

HAS_USER_SITE = True

WINDOWS_SCHEME = {
    'purelib': '{base}/Lib/site-packages',
    'platlib': '{base}/Lib/site-packages',
    'headers': '{base}/Include/{dist_name}',
    'scripts': '{base}/Scripts',
    'data': '{base}',
}

INSTALL_SCHEMES = {
    'posix_prefix': {
        'purelib': '{base}/lib/{implementation_lower}{py_version_short}/site-packages',
        'platlib': '{platbase}/{platlibdir}/{implementation_lower}'
        '{py_version_short}/site-packages',
        'headers': '{base}/include/{implementation_lower}'
        '{py_version_short}{abiflags}/{dist_name}',
        'scripts': '{base}/bin',
        'data': '{base}',
    },
    'posix_home': {
        'purelib': '{base}/lib/{implementation_lower}',
        'platlib': '{base}/{platlibdir}/{implementation_lower}',
        'headers': '{base}/include/{implementation_lower}/{dist_name}',
        'scripts': '{base}/bin',
        'data': '{base}',
    },
    'nt': WINDOWS_SCHEME,
    'pypy': {
        'purelib': '{base}/site-packages',
        'platlib': '{base}/site-packages',
        'headers': '{base}/include/{dist_name}',
        'scripts': '{base}/bin',
        'data': '{base}',
    },
    'pypy_nt': {
        'purelib': '{base}/site-packages',
        'platlib': '{base}/site-packages',
        'headers': '{base}/include/{dist_name}',
        'scripts': '{base}/Scripts',
        'data': '{base}',
    },
}

# user site schemes
if HAS_USER_SITE:
    INSTALL_SCHEMES['nt_user'] = {
        'purelib': '{usersite}',
        'platlib': '{usersite}',
        'headers': '{userbase}/{implementation}{py_version_nodot_plat}'
        '/Include/{dist_name}',
        'scripts': '{userbase}/{implementation}{py_version_nodot_plat}/Scripts',
        'data': '{userbase}',
    }

    INSTALL_SCHEMES['posix_user'] = {
        'purelib': '{usersite}',
        'platlib': '{usersite}',
        'headers': '{userbase}/include/{implementation_lower}'
        '{py_version_short}{abiflags}/{dist_name}',
        'scripts': '{userbase}/bin',
        'data': '{userbase}',
    }


INSTALL_SCHEMES.update(fw.schemes)


# The keys to an installation scheme; if any new types of files are to be
# installed, be sure to add an entry to every installation scheme above,
# and to SCHEME_KEYS here.
SCHEME_KEYS = ('purelib', 'platlib', 'headers', 'scripts', 'data')


def _load_sysconfig_schemes():
    with contextlib.suppress(AttributeError):
        return {
            scheme: sysconfig.get_paths(scheme, expand=False)
            for scheme in sysconfig.get_scheme_names()
        }


def _load_schemes():
    """
    Extend default schemes with schemes from sysconfig.
    """

    sysconfig_schemes = _load_sysconfig_schemes() or {}

    return {
        scheme: {
            **INSTALL_SCHEMES.get(scheme, {}),
            **sysconfig_schemes.get(scheme, {}),
        }
        for scheme in set(itertools.chain(INSTALL_SCHEMES, sysconfig_schemes))
    }


def _get_implementation():
    if hasattr(sys, 'pypy_version_info'):
        return 'PyPy'
    else:
        return 'Python'


def _select_scheme(ob, name):
    scheme = _inject_headers(name, _load_scheme(_resolve_scheme(name)))
    vars(ob).update(_remove_set(ob, _scheme_attrs(scheme)))


def _remove_set(ob, attrs):
    """
    Include only attrs that are None in ob.
    """
    return {key: value for key, value in attrs.items() if getattr(ob, key) is None}


def _resolve_scheme(name):
    os_name, sep, key = name.partition('_')
    try:
        resolved = sysconfig.get_preferred_scheme(key)
    except Exception:
        resolved = fw.scheme(name)
    return resolved


def _load_scheme(name):
    return _load_schemes()[name]


def _inject_headers(name, scheme):
    """
    Given a scheme name and the resolved scheme,
    if the scheme does not include headers, resolve
    the fallback scheme for the name and use headers
    from it. pypa/distutils#88
    """
    # Bypass the preferred scheme, which may not
    # have defined headers.
    fallback = _load_scheme(name)
    scheme.setdefault('headers', fallback['headers'])
    return scheme


def _scheme_attrs(scheme):
    """Resolve install directories by applying the install schemes."""
    return {f'install_{key}': scheme[key] for key in SCHEME_KEYS}


class install(Command):
    description = "install everything from build directory"

    user_options = [
        # Select installation scheme and set base director(y|ies)
        ('prefix=', None, "installation prefix"),
        ('exec-prefix=', None, "(Unix only) prefix for platform-specific files"),
        ('home=', None, "(Unix only) home directory to install under"),
        # Or, just set the base director(y|ies)
        (
            'install-base=',
            None,
            "base installation directory (instead of --prefix or --home)",
        ),
        (
            'install-platbase=',
            None,
            "base installation directory for platform-specific files (instead of --exec-prefix or --home)",
        ),
        ('root=', None, "install everything relative to this alternate root directory"),
        # Or, explicitly set the installation scheme
        (
            'install-purelib=',
            None,
            "installation directory for pure Python module distributions",
        ),
        (
            'install-platlib=',
            None,
            "installation directory for non-pure module distributions",
        ),
        (
            'install-lib=',
            None,
            "installation directory for all module distributions (overrides --install-purelib and --install-platlib)",
        ),
        ('install-headers=', None, "installation directory for C/C++ headers"),
        ('install-scripts=', None, "installation directory for Python scripts"),
        ('install-data=', None, "installation directory for data files"),
        # Byte-compilation options -- see install_lib.py for details, as
        # these are duplicated from there (but only install_lib does
        # anything with them).
        ('compile', 'c', "compile .py to .pyc [default]"),
        ('no-compile', None, "don't compile .py files"),
        (
            'optimize=',
            'O',
            "also compile with optimization: -O1 for \"python -O\", "
            "-O2 for \"python -OO\", and -O0 to disable [default: -O0]",
        ),
        # Miscellaneous control options
        ('force', 'f', "force installation (overwrite any existing files)"),
        ('skip-build', None, "skip rebuilding everything (for testing/debugging)"),
        # Where to install documentation (eventually!)
        # ('doc-format=', None, "format of documentation to generate"),
        # ('install-man=', None, "directory for Unix man pages"),
        # ('install-html=', None, "directory for HTML documentation"),
        # ('install-info=', None, "directory for GNU info files"),
        ('record=', None, "filename in which to record list of installed files"),
    ]

    boolean_options: ClassVar[list[str]] = ['compile', 'force', 'skip-build']

    if HAS_USER_SITE:
        user_options.append((
            'user',
            None,
            f"install in user site-package '{USER_SITE}'",
        ))
        boolean_options.append('user')

    negative_opt: ClassVar[dict[str, str]] = {'no-compile': 'compile'}

    def initialize_options(self) -> None:
        """Initializes options."""
        # High-level options: these select both an installation base
        # and scheme.
        self.prefix: str | None = None
        self.exec_prefix: str | None = None
        self.home: str | None = None
        self.user = False

        # These select only the installation base; it's up to the user to
        # specify the installation scheme (currently, that means supplying
        # the --install-{platlib,purelib,scripts,data} options).
        self.install_base = None
        self.install_platbase = None
        self.root: str | None = None

        # These options are the actual installation directories; if not
        # supplied by the user, they are filled in using the installation
        # scheme implied by prefix/exec-prefix/home and the contents of
        # that installation scheme.
        self.install_purelib = None  # for pure module distributions
        self.install_platlib = None  # non-pure (dists w/ extensions)
        self.install_headers = None  # for C/C++ headers
        self.install_lib: str | None = None  # set to either purelib or platlib
        self.install_scripts = None
        self.install_data = None
        self.install_userbase = USER_BASE
        self.install_usersite = USER_SITE

        self.compile = None
        self.optimize = None

        # Deprecated
        # These two are for putting non-packagized distributions into their
        # own directory and creating a .pth file if it makes sense.
        # 'extra_path' comes from the setup file; 'install_path_file' can
        # be turned off if it makes no sense to install a .pth file.  (But
        # better to install it uselessly than to guess wrong and not
        # install it when it's necessary and would be used!)  Currently,
        # 'install_path_file' is always true unless some outsider meddles
        # with it.
        self.extra_path = None
        self.install_path_file = True

        # 'force' forces installation, even if target files are not
        # out-of-date.  'skip_build' skips running the "build" command,
        # handy if you know it's not necessary.  'warn_dir' (which is *not*
        # a user option, it's just there so the bdist_* commands can turn
        # it off) determines whether we warn about installing to a
        # directory not in sys.path.
        self.force = False
        self.skip_build = False
        self.warn_dir = True

        # These are only here as a conduit from the 'build' command to the
        # 'install_*' commands that do the real work.  ('build_base' isn't
        # actually used anywhere, but it might be useful in future.)  They
        # are not user options, because if the user told the install
        # command where the build directory is, that wouldn't affect the
        # build command.
        self.build_base = None
        self.build_lib = None

        # Not defined yet because we don't know anything about
        # documentation yet.
        # self.install_man = None
        # self.install_html = None
        # self.install_info = None

        self.record = None

    # -- Option finalizing methods -------------------------------------
    # (This is rather more involved than for most commands,
    # because this is where the policy for installing third-
    # party Python modules on various platforms given a wide
    # array of user input is decided.  Yes, it's quite complex!)

    def finalize_options(self) -> None:  # noqa: C901
        """Finalizes options."""
        # This method (and its helpers, like 'finalize_unix()',
        # 'finalize_other()', and 'select_scheme()') is where the default
        # installation directories for modules, extension modules, and
        # anything else we care to install from a Python module
        # distribution.  Thus, this code makes a pretty important policy
        # statement about how third-party stuff is added to a Python
        # installation!  Note that the actual work of installation is done
        # by the relatively simple 'install_*' commands; they just take
        # their orders from the installation directory options determined
        # here.

        # Check for errors/inconsistencies in the options; first, stuff
        # that's wrong on any platform.

        if (self.prefix or self.exec_prefix or self.home) and (
            self.install_base or self.install_platbase
        ):
            raise DistutilsOptionError(
                "must supply either prefix/exec-prefix/home or install-base/install-platbase -- not both"
            )

        if self.home and (self.prefix or self.exec_prefix):
            raise DistutilsOptionError(
                "must supply either home or prefix/exec-prefix -- not both"
            )

        if self.user and (
            self.prefix
            or self.exec_prefix
            or self.home
            or self.install_base
            or self.install_platbase
        ):
            raise DistutilsOptionError(
                "can't combine user with prefix, "
                "exec_prefix/home, or install_(plat)base"
            )

        # Next, stuff that's wrong (or dubious) only on certain platforms.
        if os.name != "posix":
            if self.exec_prefix:
                self.warn("exec-prefix option ignored on this platform")
                self.exec_prefix = None

        # Now the interesting logic -- so interesting that we farm it out
        # to other methods.  The goal of these methods is to set the final
        # values for the install_{lib,scripts,data,...}  options, using as
        # input a heady brew of prefix, exec_prefix, home, install_base,
        # install_platbase, user-supplied versions of
        # install_{purelib,platlib,lib,scripts,data,...}, and the
        # install schemes.  Phew!

        self.dump_dirs("pre-finalize_{unix,other}")

        if os.name == 'posix':
            self.finalize_unix()
        else:
            self.finalize_other()

        self.dump_dirs("post-finalize_{unix,other}()")

        # Expand configuration variables, tilde, etc. in self.install_base
        # and self.install_platbase -- that way, we can use $base or
        # $platbase in the other installation directories and not worry
        # about needing recursive variable expansion (shudder).

        py_version = sys.version.split()[0]
        (prefix, exec_prefix) = get_config_vars('prefix', 'exec_prefix')
        try:
            abiflags = sys.abiflags
        except AttributeError:
            # sys.abiflags may not be defined on all platforms.
            abiflags = ''
        local_vars = {
            'dist_name': self.distribution.get_name(),
            'dist_version': self.distribution.get_version(),
            'dist_fullname': self.distribution.get_fullname(),
            'py_version': py_version,
            'py_version_short': f'{sys.version_info.major}.{sys.version_info.minor}',
            'py_version_nodot': f'{sys.version_info.major}{sys.version_info.minor}',
            'sys_prefix': prefix,
            'prefix': prefix,
            'sys_exec_prefix': exec_prefix,
            'exec_prefix': exec_prefix,
            'abiflags': abiflags,
            'platlibdir': getattr(sys, 'platlibdir', 'lib'),
            'implementation_lower': _get_implementation().lower(),
            'implementation': _get_implementation(),
        }

        # vars for compatibility on older Pythons
        compat_vars = dict(
            # Python 3.9 and earlier
            py_version_nodot_plat=getattr(sys, 'winver', '').replace('.', ''),
        )

        if HAS_USER_SITE:
            local_vars['userbase'] = self.install_userbase
            local_vars['usersite'] = self.install_usersite

        self.config_vars = collections.ChainMap(
            local_vars,
            sysconfig.get_config_vars(),
            compat_vars,
            fw.vars(),
        )

        self.expand_basedirs()

        self.dump_dirs("post-expand_basedirs()")

        # Now define config vars for the base directories so we can expand
        # everything else.
        local_vars['base'] = self.install_base
        local_vars['platbase'] = self.install_platbase

        if DEBUG:
            from pprint import pprint

            print("config vars:")
            pprint(dict(self.config_vars))

        # Expand "~" and configuration variables in the installation
        # directories.
        self.expand_dirs()

        self.dump_dirs("post-expand_dirs()")

        # Create directories in the home dir:
        if self.user:
            self.create_home_path()

        # Pick the actual directory to install all modules to: either
        # install_purelib or install_platlib, depending on whether this
        # module distribution is pure or not.  Of course, if the user
        # already specified install_lib, use their selection.
        if self.install_lib is None:
            if self.distribution.has_ext_modules():  # has extensions: non-pure
                self.install_lib = self.install_platlib
            else:
                self.install_lib = self.install_purelib

        # Convert directories from Unix /-separated syntax to the local
        # convention.
        self.convert_paths(
            'lib',
            'purelib',
            'platlib',
            'scripts',
            'data',
            'headers',
            'userbase',
            'usersite',
        )

        # Deprecated
        # Well, we're not actually fully completely finalized yet: we still
        # have to deal with 'extra_path', which is the hack for allowing
        # non-packagized module distributions (hello, Numerical Python!) to
        # get their own directories.
        self.handle_extra_path()
        self.install_libbase = self.install_lib  # needed for .pth file
        self.install_lib = os.path.join(self.install_lib, self.extra_dirs)

        # If a new root directory was supplied, make all the installation
        # dirs relative to it.
        if self.root is not None:
            self.change_roots(
                'libbase', 'lib', 'purelib', 'platlib', 'scripts', 'data', 'headers'
            )

        self.dump_dirs("after prepending root")

        # Find out the build directories, ie. where to install from.
        self.set_undefined_options(
            'build', ('build_base', 'build_base'), ('build_lib', 'build_lib')
        )

        # Punt on doc directories for now -- after all, we're punting on
        # documentation completely!

    def dump_dirs(self, msg) -> None:
        """Dumps the list of user options."""
        if not DEBUG:
            return
        from ..fancy_getopt import longopt_xlate

        log.debug(msg + ":")
        for opt in self.user_options:
            opt_name = opt[0]
            if opt_name[-1] == "=":
                opt_name = opt_name[0:-1]
            if opt_name in self.negative_opt:
                opt_name = self.negative_opt[opt_name]
                opt_name = opt_name.translate(longopt_xlate)
                val = not getattr(self, opt_name)
            else:
                opt_name = opt_name.translate(longopt_xlate)
                val = getattr(self, opt_name)
            log.debug("  %s: %s", opt_name, val)

    def finalize_unix(self) -> None:
        """Finalizes options for posix platforms."""
        if self.install_base is not None or self.install_platbase is not None:
            incomplete_scheme = (
                (
                    self.install_lib is None
                    and self.install_purelib is None
                    and self.install_platlib is None
                )
                or self.install_headers is None
                or self.install_scripts is None
                or self.install_data is None
            )
            if incomplete_scheme:
                raise DistutilsOptionError(
                    "install-base or install-platbase supplied, but "
                    "installation scheme is incomplete"
                )
            return

        if self.user:
            if self.install_userbase is None:
                raise DistutilsPlatformError("User base directory is not specified")
            self.install_base = self.install_platbase = self.install_userbase
            self.select_scheme("posix_user")
        elif self.home is not None:
            self.install_base = self.install_platbase = self.home
            self.select_scheme("posix_home")
        else:
            if self.prefix is None:
                if self.exec_prefix is not None:
                    raise DistutilsOptionError(
                        "must not supply exec-prefix without prefix"
                    )

                # Allow Fedora to add components to the prefix
                _prefix_addition = getattr(sysconfig, '_prefix_addition', "")

                self.prefix = os.path.normpath(sys.prefix) + _prefix_addition
                self.exec_prefix = os.path.normpath(sys.exec_prefix) + _prefix_addition

            else:
                if self.exec_prefix is None:
                    self.exec_prefix = self.prefix

            self.install_base = self.prefix
            self.install_platbase = self.exec_prefix
            self.select_scheme("posix_prefix")

    def finalize_other(self) -> None:
        """Finalizes options for non-posix platforms"""
        if self.user:
            if self.install_userbase is None:
                raise DistutilsPlatformError("User base directory is not specified")
            self.install_base = self.install_platbase = self.install_userbase
            self.select_scheme(os.name + "_user")
        elif self.home is not None:
            self.install_base = self.install_platbase = self.home
            self.select_scheme("posix_home")
        else:
            if self.prefix is None:
                self.prefix = os.path.normpath(sys.prefix)

            self.install_base = self.install_platbase = self.prefix
            try:
                self.select_scheme(os.name)
            except KeyError:
                raise DistutilsPlatformError(
                    f"I don't know how to install stuff on '{os.name}'"
                )

    def select_scheme(self, name) -> None:
        _select_scheme(self, name)

    def _expand_attrs(self, attrs):
        for attr in attrs:
            val = getattr(self, attr)
            if val is not None:
                if os.name in ('posix', 'nt'):
                    val = os.path.expanduser(val)
                val = subst_vars(val, self.config_vars)
                setattr(self, attr, val)

    def expand_basedirs(self) -> None:
        """Calls `os.path.expanduser` on install_base, install_platbase and
        root."""
        self._expand_attrs(['install_base', 'install_platbase', 'root'])

    def expand_dirs(self) -> None:
        """Calls `os.path.expanduser` on install dirs."""
        self._expand_attrs([
            'install_purelib',
            'install_platlib',
            'install_lib',
            'install_headers',
            'install_scripts',
            'install_data',
        ])

    def convert_paths(self, *names) -> None:
        """Call `convert_path` over `names`."""
        for name in names:
            attr = "install_" + name
            setattr(self, attr, convert_path(getattr(self, attr)))

    def handle_extra_path(self) -> None:
        """Set `path_file` and `extra_dirs` using `extra_path`."""
        if self.extra_path is None:
            self.extra_path = self.distribution.extra_path

        if self.extra_path is not None:
            log.warning(
                "Distribution option extra_path is deprecated. "
                "See issue27919 for details."
            )
            if isinstance(self.extra_path, str):
                self.extra_path = self.extra_path.split(',')

            if len(self.extra_path) == 1:
                path_file = extra_dirs = self.extra_path[0]
            elif len(self.extra_path) == 2:
                path_file, extra_dirs = self.extra_path
            else:
                raise DistutilsOptionError(
                    "'extra_path' option must be a list, tuple, or "
                    "comma-separated string with 1 or 2 elements"
                )

            # convert to local form in case Unix notation used (as it
            # should be in setup scripts)
            extra_dirs = convert_path(extra_dirs)
        else:
            path_file = None
            extra_dirs = ''

        # XXX should we warn if path_file and not extra_dirs? (in which
        # case the path file would be harmless but pointless)
        self.path_file = path_file
        self.extra_dirs = extra_dirs

    def change_roots(self, *names) -> None:
        """Change the install directories pointed by name using root."""
        for name in names:
            attr = "install_" + name
            setattr(self, attr, change_root(self.root, getattr(self, attr)))

    def create_home_path(self) -> None:
        """Create directories under ~."""
        if not self.user:
            return
        home = convert_path(os.path.expanduser("~"))
        for path in self.config_vars.values():
            if str(path).startswith(home) and not os.path.isdir(path):
                self.debug_print(f"os.makedirs('{path}', 0o700)")
                os.makedirs(path, 0o700)

    # -- Command execution methods -------------------------------------

    def run(self):
        """Runs the command."""
        # Obviously have to build before we can install
        if not self.skip_build:
            self.run_command('build')
            # If we built for any other platform, we can't install.
            build_plat = self.distribution.get_command_obj('build').plat_name
            # check warn_dir - it is a clue that the 'install' is happening
            # internally, and not to sys.path, so we don't check the platform
            # matches what we are running.
            if self.warn_dir and build_plat != get_platform():
                raise DistutilsPlatformError("Can't install when cross-compiling")

        # Run all sub-commands (at least those that need to be run)
        for cmd_name in self.get_sub_commands():
            self.run_command(cmd_name)

        if self.path_file:
            self.create_path_file()

        # write list of installed files, if requested.
        if self.record:
            outputs = self.get_outputs()
            if self.root:  # strip any package prefix
                root_len = len(self.root)
                for counter in range(len(outputs)):
                    outputs[counter] = outputs[counter][root_len:]
            self.execute(
                write_file,
                (self.record, outputs),
                f"writing list of installed files to '{self.record}'",
            )

        sys_path = map(os.path.normpath, sys.path)
        sys_path = map(os.path.normcase, sys_path)
        install_lib = os.path.normcase(os.path.normpath(self.install_lib))
        if (
            self.warn_dir
            and not (self.path_file and self.install_path_file)
            and install_lib not in sys_path
        ):
            log.debug(
                (
                    "modules installed to '%s', which is not in "
                    "Python's module search path (sys.path) -- "
                    "you'll have to change the search path yourself"
                ),
                self.install_lib,
            )

    def create_path_file(self):
        """Creates the .pth file"""
        filename = os.path.join(self.install_libbase, self.path_file + ".pth")
        if self.install_path_file:
            self.execute(
                write_file, (filename, [self.extra_dirs]), f"creating {filename}"
            )
        else:
            self.warn(f"path file '{filename}' not created")

    # -- Reporting methods ---------------------------------------------

    def get_outputs(self):
        """Assembles the outputs of all the sub-commands."""
        outputs = []
        for cmd_name in self.get_sub_commands():
            cmd = self.get_finalized_command(cmd_name)
            # Add the contents of cmd.get_outputs(), ensuring
            # that outputs doesn't contain duplicate entries
            for filename in cmd.get_outputs():
                if filename not in outputs:
                    outputs.append(filename)

        if self.path_file and self.install_path_file:
            outputs.append(os.path.join(self.install_libbase, self.path_file + ".pth"))

        return outputs

    def get_inputs(self):
        """Returns the inputs of all the sub-commands"""
        # XXX gee, this looks familiar ;-(
        inputs = []
        for cmd_name in self.get_sub_commands():
            cmd = self.get_finalized_command(cmd_name)
            inputs.extend(cmd.get_inputs())

        return inputs

    # -- Predicates for sub-command list -------------------------------

    def has_lib(self):
        """Returns true if the current distribution has any Python
        modules to install."""
        return (
            self.distribution.has_pure_modules() or self.distribution.has_ext_modules()
        )

    def has_headers(self):
        """Returns true if the current distribution has any headers to
        install."""
        return self.distribution.has_headers()

    def has_scripts(self):
        """Returns true if the current distribution has any scripts to.
        install."""
        return self.distribution.has_scripts()

    def has_data(self):
        """Returns true if the current distribution has any data to.
        install."""
        return self.distribution.has_data_files()

    # 'sub_commands': a list of commands this command might have to run to
    # get its work done.  See cmd.py for more info.
    sub_commands = [
        ('install_lib', has_lib),
        ('install_headers', has_headers),
        ('install_scripts', has_scripts),
        ('install_data', has_data),
        ('install_egg_info', lambda self: True),
    ]
