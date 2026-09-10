from __future__ import annotations

import os
import sys

from .._path import ensure_directory
from ..dist import Distribution

import distutils.command.install_scripts as orig
from distutils import log


class install_scripts(orig.install_scripts):
    """Do normal script install, plus any egg_info wrapper scripts"""

    distribution: Distribution  # override distutils.dist.Distribution with setuptools.dist.Distribution

    def initialize_options(self) -> None:
        orig.install_scripts.initialize_options(self)
        self.no_ep = False

    def run(self) -> None:
        self.run_command("egg_info")
        if self.distribution.scripts:
            orig.install_scripts.run(self)  # run first to set up self.outfiles
        else:
            self.outfiles: list[str] = []
        if self.no_ep:
            # don't install entry point scripts into .egg file!
            return
        self._install_ep_scripts()

    def _install_ep_scripts(self):
        # Delay import side-effects
        from pkg_resources import Distribution, PathMetadata

        from . import easy_install as ei

        ei_cmd = self.get_finalized_command("egg_info")
        dist = Distribution(
            ei_cmd.egg_base,
            PathMetadata(ei_cmd.egg_base, ei_cmd.egg_info),
            ei_cmd.egg_name,
            ei_cmd.egg_version,
        )
        bs_cmd = self.get_finalized_command('build_scripts')
        exec_param = getattr(bs_cmd, 'executable', None)
        writer = ei.ScriptWriter
        if exec_param == sys.executable:
            # In case the path to the Python executable contains a space, wrap
            # it so it's not split up.
            exec_param = [exec_param]
        # resolve the writer to the environment
        writer = writer.best()
        cmd = writer.command_spec_class.best().from_param(exec_param)
        for args in writer.get_args(dist, cmd.as_header()):
            self.write_script(*args)

    def write_script(self, script_name, contents, mode: str = "t", *ignored) -> None:
        """Write an executable file to the scripts directory"""
        from setuptools.command.easy_install import chmod, current_umask

        log.info("Installing %s script to %s", script_name, self.install_dir)
        target = os.path.join(self.install_dir, script_name)
        self.outfiles.append(target)

        encoding = None if "b" in mode else "utf-8"
        mask = current_umask()
        if not self.dry_run:
            ensure_directory(target)
            with open(target, "w" + mode, encoding=encoding) as f:
                f.write(contents)
            chmod(target, 0o777 - mask)
