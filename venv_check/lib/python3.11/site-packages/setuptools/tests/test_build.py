from setuptools import Command
from setuptools.command.build import build
from setuptools.dist import Distribution


def test_distribution_gives_setuptools_build_obj(tmpdir_cwd):
    """
    Check that the setuptools Distribution uses the
    setuptools specific build object.
    """

    dist = Distribution(
        dict(
            script_name='setup.py',
            script_args=['build'],
            packages=[],
            package_data={'': ['path/*']},
        )
    )
    assert isinstance(dist.get_command_obj("build"), build)


class Subcommand(Command):
    """Dummy command to be used in tests"""

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        raise NotImplementedError("just to check if the command runs")
