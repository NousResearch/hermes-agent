from __future__ import annotations

import glob
import os
import subprocess
import sys
import tempfile
from functools import partial

from pkg_resources import Distribution

from . import _reqs
from ._reqs import _StrOrIter
from .warnings import SetuptoolsDeprecationWarning
from .wheel import Wheel

from distutils import log
from distutils.errors import DistutilsError


def _fixup_find_links(find_links):
    """Ensure find-links option end-up being a list of strings."""
    if isinstance(find_links, str):
        return find_links.split()
    assert isinstance(find_links, (tuple, list))
    return find_links


def fetch_build_egg(dist, req):
    """Fetch an egg needed for building.

    Use pip/wheel to fetch/build a wheel."""
    _DeprecatedInstaller.emit()
    _warn_wheel_not_available(dist)
    return _fetch_build_egg_no_warn(dist, req)


def _fetch_build_eggs(dist, requires: _StrOrIter) -> list[Distribution]:
    import pkg_resources  # Delay import to avoid unnecessary side-effects

    _DeprecatedInstaller.emit(stacklevel=3)
    _warn_wheel_not_available(dist)

    resolved_dists = pkg_resources.working_set.resolve(
        _reqs.parse(requires, pkg_resources.Requirement),  # required for compatibility
        installer=partial(_fetch_build_egg_no_warn, dist),  # avoid warning twice
        replace_conflicting=True,
    )
    for dist in resolved_dists:
        pkg_resources.working_set.add(dist, replace=True)
    return resolved_dists


def _fetch_build_egg_no_warn(dist, req):  # noqa: C901  # is too complex (16)  # FIXME
    import pkg_resources  # Delay import to avoid unnecessary side-effects

    # Ignore environment markers; if supplied, it is required.
    req = strip_marker(req)
    # Take easy_install options into account, but do not override relevant
    # pip environment variables (like PIP_INDEX_URL or PIP_QUIET); they'll
    # take precedence.
    opts = dist.get_option_dict('easy_install')
    if 'allow_hosts' in opts:
        raise DistutilsError(
            'the `allow-hosts` option is not supported '
            'when using pip to install requirements.'
        )
    quiet = 'PIP_QUIET' not in os.environ and 'PIP_VERBOSE' not in os.environ
    if 'PIP_INDEX_URL' in os.environ:
        index_url = None
    elif 'index_url' in opts:
        index_url = opts['index_url'][1]
    else:
        index_url = None
    find_links = (
        _fixup_find_links(opts['find_links'][1])[:] if 'find_links' in opts else []
    )
    if dist.dependency_links:
        find_links.extend(dist.dependency_links)
    eggs_dir = os.path.realpath(dist.get_egg_cache_dir())
    environment = pkg_resources.Environment()
    for egg_dist in pkg_resources.find_distributions(eggs_dir):
        if egg_dist in req and environment.can_add(egg_dist):
            return egg_dist
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable,
            '-m',
            'pip',
            '--disable-pip-version-check',
            'wheel',
            '--no-deps',
            '-w',
            tmpdir,
        ]
        if quiet:
            cmd.append('--quiet')
        if index_url is not None:
            cmd.extend(('--index-url', index_url))
        for link in find_links or []:
            cmd.extend(('--find-links', link))
        # If requirement is a PEP 508 direct URL, directly pass
        # the URL to pip, as `req @ url` does not work on the
        # command line.
        cmd.append(req.url or str(req))
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            raise DistutilsError(str(e)) from e
        wheel = Wheel(glob.glob(os.path.join(tmpdir, '*.whl'))[0])
        dist_location = os.path.join(eggs_dir, wheel.egg_name())
        wheel.install_as_egg(dist_location)
        dist_metadata = pkg_resources.PathMetadata(
            dist_location, os.path.join(dist_location, 'EGG-INFO')
        )
        return pkg_resources.Distribution.from_filename(
            dist_location, metadata=dist_metadata
        )


def strip_marker(req):
    """
    Return a new requirement without the environment marker to avoid
    calling pip with something like `babel; extra == "i18n"`, which
    would always be ignored.
    """
    import pkg_resources  # Delay import to avoid unnecessary side-effects

    # create a copy to avoid mutating the input
    req = pkg_resources.Requirement.parse(str(req))
    req.marker = None
    return req


def _warn_wheel_not_available(dist):
    import pkg_resources  # Delay import to avoid unnecessary side-effects

    try:
        pkg_resources.get_distribution('wheel')
    except pkg_resources.DistributionNotFound:
        dist.announce('WARNING: The wheel package is not available.', log.WARN)


class _DeprecatedInstaller(SetuptoolsDeprecationWarning):
    _SUMMARY = "setuptools.installer and fetch_build_eggs are deprecated."
    _DETAILS = """
    Requirements should be satisfied by a PEP 517 installer.
    If you are using pip, you can try `pip install --use-pep517`.
    """
    # _DUE_DATE not decided yet
