import glob
import os

import setuptools
from setuptools import _normalization, _path, namespaces
from setuptools.command.easy_install import easy_install

from ..unicode_utils import _read_utf8_with_fallback

from distutils import log
from distutils.errors import DistutilsOptionError
from distutils.util import convert_path


class develop(namespaces.DevelopInstaller, easy_install):
    """Set up package for development"""

    description = "install package in 'development mode'"

    user_options = easy_install.user_options + [
        ("uninstall", "u", "Uninstall this source package"),
        ("egg-path=", None, "Set the path to be used in the .egg-link file"),
    ]

    boolean_options = easy_install.boolean_options + ['uninstall']

    command_consumes_arguments = False  # override base

    def run(self):
        if self.uninstall:
            self.multi_version = True
            self.uninstall_link()
            self.uninstall_namespaces()
        else:
            self.install_for_development()
        self.warn_deprecated_options()

    def initialize_options(self):
        self.uninstall = None
        self.egg_path = None
        easy_install.initialize_options(self)
        self.setup_path = None
        self.always_copy_from = '.'  # always copy eggs installed in curdir

    def finalize_options(self) -> None:
        import pkg_resources

        ei = self.get_finalized_command("egg_info")
        self.args = [ei.egg_name]

        easy_install.finalize_options(self)
        self.expand_basedirs()
        self.expand_dirs()
        # pick up setup-dir .egg files only: no .egg-info
        self.package_index.scan(glob.glob('*.egg'))

        egg_link_fn = (
            _normalization.filename_component_broken(ei.egg_name) + '.egg-link'
        )
        self.egg_link = os.path.join(self.install_dir, egg_link_fn)
        self.egg_base = ei.egg_base
        if self.egg_path is None:
            self.egg_path = os.path.abspath(ei.egg_base)

        target = _path.normpath(self.egg_base)
        egg_path = _path.normpath(os.path.join(self.install_dir, self.egg_path))
        if egg_path != target:
            raise DistutilsOptionError(
                "--egg-path must be a relative path from the install"
                " directory to " + target
            )

        # Make a distribution for the package's source
        self.dist = pkg_resources.Distribution(
            target,
            pkg_resources.PathMetadata(target, os.path.abspath(ei.egg_info)),
            project_name=ei.egg_name,
        )

        self.setup_path = self._resolve_setup_path(
            self.egg_base,
            self.install_dir,
            self.egg_path,
        )

    @staticmethod
    def _resolve_setup_path(egg_base, install_dir, egg_path):
        """
        Generate a path from egg_base back to '.' where the
        setup script resides and ensure that path points to the
        setup path from $install_dir/$egg_path.
        """
        path_to_setup = egg_base.replace(os.sep, '/').rstrip('/')
        if path_to_setup != os.curdir:
            path_to_setup = '../' * (path_to_setup.count('/') + 1)
        resolved = _path.normpath(os.path.join(install_dir, egg_path, path_to_setup))
        curdir = _path.normpath(os.curdir)
        if resolved != curdir:
            raise DistutilsOptionError(
                "Can't get a consistent path to setup script from"
                " installation directory",
                resolved,
                curdir,
            )
        return path_to_setup

    def install_for_development(self) -> None:
        self.run_command('egg_info')

        # Build extensions in-place
        self.reinitialize_command('build_ext', inplace=True)
        self.run_command('build_ext')

        if setuptools.bootstrap_install_from:
            self.easy_install(setuptools.bootstrap_install_from)
            setuptools.bootstrap_install_from = None

        self.install_namespaces()

        # create an .egg-link in the installation dir, pointing to our egg
        log.info("Creating %s (link to %s)", self.egg_link, self.egg_base)
        if not self.dry_run:
            with open(self.egg_link, "w", encoding="utf-8") as f:
                f.write(self.egg_path + "\n" + self.setup_path)
        # postprocess the installed distro, fixing up .pth, installing scripts,
        # and handling requirements
        self.process_distribution(None, self.dist, not self.no_deps)

    def uninstall_link(self) -> None:
        if os.path.exists(self.egg_link):
            log.info("Removing %s (link to %s)", self.egg_link, self.egg_base)

            contents = [
                line.rstrip()
                for line in _read_utf8_with_fallback(self.egg_link).splitlines()
            ]

            if contents not in ([self.egg_path], [self.egg_path, self.setup_path]):
                log.warn("Link points to %s: uninstall aborted", contents)
                return
            if not self.dry_run:
                os.unlink(self.egg_link)
        if not self.dry_run:
            self.update_pth(self.dist)  # remove any .pth link to us
        if self.distribution.scripts:
            # XXX should also check for entry point scripts!
            log.warn("Note: you must uninstall or replace scripts manually!")

    def install_egg_scripts(self, dist):
        if dist is not self.dist:
            # Installing a dependency, so fall back to normal behavior
            return easy_install.install_egg_scripts(self, dist)

        # create wrapper scripts in the script dir, pointing to dist.scripts

        # new-style...
        self.install_wrapper_scripts(dist)

        # ...and old-style
        for script_name in self.distribution.scripts or []:
            script_path = os.path.abspath(convert_path(script_name))
            script_name = os.path.basename(script_path)
            script_text = _read_utf8_with_fallback(script_path)
            self.install_script(dist, script_name, script_text, script_path)

        return None

    def install_wrapper_scripts(self, dist):
        dist = VersionlessRequirement(dist)
        return easy_install.install_wrapper_scripts(self, dist)


class VersionlessRequirement:
    """
    Adapt a pkg_resources.Distribution to simply return the project
    name as the 'requirement' so that scripts will work across
    multiple versions.

    >>> from pkg_resources import Distribution
    >>> dist = Distribution(project_name='foo', version='1.0')
    >>> str(dist.as_requirement())
    'foo==1.0'
    >>> adapted_dist = VersionlessRequirement(dist)
    >>> str(adapted_dist.as_requirement())
    'foo'
    """

    def __init__(self, dist) -> None:
        self.__dist = dist

    def __getattr__(self, name: str):
        return getattr(self.__dist, name)

    def as_requirement(self):
        return self.project_name
