import subprocess
import sys

from setuptools._path import paths_on_pythonpath

from . import namespaces


class TestNamespaces:
    def test_mixed_site_and_non_site(self, tmpdir):
        """
        Installing two packages sharing the same namespace, one installed
        to a site dir and the other installed just to a path on PYTHONPATH
        should leave the namespace in tact and both packages reachable by
        import.
        """
        pkg_A = namespaces.build_namespace_package(tmpdir, 'myns.pkgA')
        pkg_B = namespaces.build_namespace_package(tmpdir, 'myns.pkgB')
        site_packages = tmpdir / 'site-packages'
        path_packages = tmpdir / 'path-packages'
        targets = site_packages, path_packages
        # use pip to install to the target directory
        install_cmd = [
            sys.executable,
            '-m',
            'pip.__main__',
            'install',
            str(pkg_A),
            '-t',
            str(site_packages),
        ]
        subprocess.check_call(install_cmd)
        namespaces.make_site_dir(site_packages)
        install_cmd = [
            sys.executable,
            '-m',
            'pip.__main__',
            'install',
            str(pkg_B),
            '-t',
            str(path_packages),
        ]
        subprocess.check_call(install_cmd)
        try_import = [
            sys.executable,
            '-c',
            'import myns.pkgA; import myns.pkgB',
        ]
        with paths_on_pythonpath(map(str, targets)):
            subprocess.check_call(try_import)

    def test_pkg_resources_import(self, tmpdir):
        """
        Ensure that a namespace package doesn't break on import
        of pkg_resources.
        """
        pkg = namespaces.build_namespace_package(tmpdir, 'myns.pkgA')
        target = tmpdir / 'packages'
        target.mkdir()
        install_cmd = [
            sys.executable,
            '-m',
            'pip',
            'install',
            '-t',
            str(target),
            str(pkg),
        ]
        with paths_on_pythonpath([str(target)]):
            subprocess.check_call(install_cmd)
        namespaces.make_site_dir(target)
        try_import = [
            sys.executable,
            '-c',
            'import pkg_resources',
        ]
        with paths_on_pythonpath([str(target)]):
            subprocess.check_call(try_import)

    def test_namespace_package_installed_and_cwd(self, tmpdir):
        """
        Installing a namespace packages but also having it in the current
        working directory, only one version should take precedence.
        """
        pkg_A = namespaces.build_namespace_package(tmpdir, 'myns.pkgA')
        target = tmpdir / 'packages'
        # use pip to install to the target directory
        install_cmd = [
            sys.executable,
            '-m',
            'pip.__main__',
            'install',
            str(pkg_A),
            '-t',
            str(target),
        ]
        subprocess.check_call(install_cmd)
        namespaces.make_site_dir(target)

        # ensure that package imports and pkg_resources imports
        pkg_resources_imp = [
            sys.executable,
            '-c',
            'import pkg_resources; import myns.pkgA',
        ]
        with paths_on_pythonpath([str(target)]):
            subprocess.check_call(pkg_resources_imp, cwd=str(pkg_A))

    def test_packages_in_the_same_namespace_installed_and_cwd(self, tmpdir):
        """
        Installing one namespace package and also have another in the same
        namespace in the current working directory, both of them must be
        importable.
        """
        pkg_A = namespaces.build_namespace_package(tmpdir, 'myns.pkgA')
        pkg_B = namespaces.build_namespace_package(tmpdir, 'myns.pkgB')
        target = tmpdir / 'packages'
        # use pip to install to the target directory
        install_cmd = [
            sys.executable,
            '-m',
            'pip.__main__',
            'install',
            str(pkg_A),
            '-t',
            str(target),
        ]
        subprocess.check_call(install_cmd)
        namespaces.make_site_dir(target)

        # ensure that all packages import and pkg_resources imports
        pkg_resources_imp = [
            sys.executable,
            '-c',
            'import pkg_resources; import myns.pkgA; import myns.pkgB',
        ]
        with paths_on_pythonpath([str(target)]):
            subprocess.check_call(pkg_resources_imp, cwd=str(pkg_B))
