"""distutils.command.install_data

Implements the Distutils 'install_data' command, for installing
platform-independent data files."""

# contributed by Bastian Kleineidam

from __future__ import annotations

import functools
import os
from collections.abc import Iterable
from typing import ClassVar

from ..core import Command
from ..util import change_root, convert_path


class install_data(Command):
    description = "install data files"

    user_options = [
        (
            'install-dir=',
            'd',
            "base directory for installing data files [default: installation base dir]",
        ),
        ('root=', None, "install everything relative to this alternate root directory"),
        ('force', 'f', "force installation (overwrite existing files)"),
    ]

    boolean_options: ClassVar[list[str]] = ['force']

    def initialize_options(self):
        self.install_dir = None
        self.outfiles = []
        self.root = None
        self.force = False
        self.data_files = self.distribution.data_files
        self.warn_dir = True

    def finalize_options(self) -> None:
        self.set_undefined_options(
            'install',
            ('install_data', 'install_dir'),
            ('root', 'root'),
            ('force', 'force'),
        )

    def run(self) -> None:
        self.mkpath(self.install_dir)
        for f in self.data_files:
            self._copy(f)

    @functools.singledispatchmethod
    def _copy(self, f: tuple[str | os.PathLike, Iterable[str | os.PathLike]]):
        # it's a tuple with path to install to and a list of files
        dir = convert_path(f[0])
        if not os.path.isabs(dir):
            dir = os.path.join(self.install_dir, dir)
        elif self.root:
            dir = change_root(self.root, dir)
        self.mkpath(dir)

        if f[1] == []:
            # If there are no files listed, the user must be
            # trying to create an empty directory, so add the
            # directory to the list of output files.
            self.outfiles.append(dir)
        else:
            # Copy files, adding them to the list of output files.
            for data in f[1]:
                data = convert_path(data)
                (out, _) = self.copy_file(data, dir)
                self.outfiles.append(out)

    @_copy.register(str)
    @_copy.register(os.PathLike)
    def _(self, f: str | os.PathLike):
        # it's a simple file, so copy it
        f = convert_path(f)
        if self.warn_dir:
            self.warn(
                "setup script did not provide a directory for "
                f"'{f}' -- installing right in '{self.install_dir}'"
            )
        (out, _) = self.copy_file(f, self.install_dir)
        self.outfiles.append(out)

    def get_inputs(self):
        return self.data_files or []

    def get_outputs(self):
        return self.outfiles
