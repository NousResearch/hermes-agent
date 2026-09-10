"""
distutils.command.install_egg_info

Implements the Distutils 'install_egg_info' command, for installing
a package's PKG-INFO metadata.
"""

import os
import re
import sys
from typing import ClassVar

from .. import dir_util
from .._log import log
from ..cmd import Command


class install_egg_info(Command):
    """Install an .egg-info file for the package"""

    description = "Install package's PKG-INFO metadata as an .egg-info file"
    user_options: ClassVar[list[tuple[str, str, str]]] = [
        ('install-dir=', 'd', "directory to install to"),
    ]

    def initialize_options(self):
        self.install_dir = None

    @property
    def basename(self):
        """
        Allow basename to be overridden by child class.
        Ref pypa/distutils#2.
        """
        name = to_filename(safe_name(self.distribution.get_name()))
        version = to_filename(safe_version(self.distribution.get_version()))
        return f"{name}-{version}-py{sys.version_info.major}.{sys.version_info.minor}.egg-info"

    def finalize_options(self):
        self.set_undefined_options('install_lib', ('install_dir', 'install_dir'))
        self.target = os.path.join(self.install_dir, self.basename)
        self.outputs = [self.target]

    def run(self):
        target = self.target
        if os.path.isdir(target) and not os.path.islink(target):
            dir_util.remove_tree(target, dry_run=self.dry_run)
        elif os.path.exists(target):
            self.execute(os.unlink, (self.target,), "Removing " + target)
        elif not os.path.isdir(self.install_dir):
            self.execute(
                os.makedirs, (self.install_dir,), "Creating " + self.install_dir
            )
        log.info("Writing %s", target)
        if not self.dry_run:
            with open(target, 'w', encoding='UTF-8') as f:
                self.distribution.metadata.write_pkg_file(f)

    def get_outputs(self):
        return self.outputs


# The following routines are taken from setuptools' pkg_resources module and
# can be replaced by importing them from pkg_resources once it is included
# in the stdlib.


def safe_name(name):
    """Convert an arbitrary string to a standard distribution name

    Any runs of non-alphanumeric/. characters are replaced with a single '-'.
    """
    return re.sub('[^A-Za-z0-9.]+', '-', name)


def safe_version(version):
    """Convert an arbitrary string to a standard version string

    Spaces become dots, and all other non-alphanumeric characters become
    dashes, with runs of multiple dashes condensed to a single dash.
    """
    version = version.replace(' ', '.')
    return re.sub('[^A-Za-z0-9.]+', '-', version)


def to_filename(name):
    """Convert a project or version name to its filename-escaped form

    Any '-' characters are currently replaced with '_'.
    """
    return name.replace('-', '_')
