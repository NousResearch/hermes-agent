"""distutils.archive_util

Utility functions for creating archive files (tarballs, zip files,
that sort of thing)."""

from __future__ import annotations

import os
from typing import Literal, overload

try:
    import zipfile
except ImportError:
    zipfile = None


from ._log import log
from .dir_util import mkpath
from .errors import DistutilsExecError
from .spawn import spawn

try:
    from pwd import getpwnam
except ImportError:
    getpwnam = None

try:
    from grp import getgrnam
except ImportError:
    getgrnam = None


def _get_gid(name):
    """Returns a gid, given a group name."""
    if getgrnam is None or name is None:
        return None
    try:
        result = getgrnam(name)
    except KeyError:
        result = None
    if result is not None:
        return result[2]
    return None


def _get_uid(name):
    """Returns an uid, given a user name."""
    if getpwnam is None or name is None:
        return None
    try:
        result = getpwnam(name)
    except KeyError:
        result = None
    if result is not None:
        return result[2]
    return None


def make_tarball(
    base_name: str,
    base_dir: str | os.PathLike[str],
    compress: Literal["gzip", "bzip2", "xz"] | None = "gzip",
    verbose: bool = False,
    dry_run: bool = False,
    owner: str | None = None,
    group: str | None = None,
) -> str:
    """Create a (possibly compressed) tar file from all the files under
    'base_dir'.

    'compress' must be "gzip" (the default), "bzip2", "xz", or None.

    'owner' and 'group' can be used to define an owner and a group for the
    archive that is being built. If not provided, the current owner and group
    will be used.

    The output tar file will be named 'base_dir' +  ".tar", possibly plus
    the appropriate compression extension (".gz", ".bz2", ".xz" or ".Z").

    Returns the output filename.
    """
    tar_compression = {
        'gzip': 'gz',
        'bzip2': 'bz2',
        'xz': 'xz',
        None: '',
    }
    compress_ext = {'gzip': '.gz', 'bzip2': '.bz2', 'xz': '.xz'}

    # flags for compression program, each element of list will be an argument
    if compress is not None and compress not in compress_ext.keys():
        raise ValueError(
            "bad value for 'compress': must be None, 'gzip', 'bzip2', 'xz'"
        )

    archive_name = base_name + '.tar'
    archive_name += compress_ext.get(compress, '')

    mkpath(os.path.dirname(archive_name), dry_run=dry_run)

    # creating the tarball
    import tarfile  # late import so Python build itself doesn't break

    log.info('Creating tar archive')

    uid = _get_uid(owner)
    gid = _get_gid(group)

    def _set_uid_gid(tarinfo):
        if gid is not None:
            tarinfo.gid = gid
            tarinfo.gname = group
        if uid is not None:
            tarinfo.uid = uid
            tarinfo.uname = owner
        return tarinfo

    if not dry_run:
        tar = tarfile.open(archive_name, f'w|{tar_compression[compress]}')
        try:
            tar.add(base_dir, filter=_set_uid_gid)
        finally:
            tar.close()

    return archive_name


def make_zipfile(  # noqa: C901
    base_name: str,
    base_dir: str | os.PathLike[str],
    verbose: bool = False,
    dry_run: bool = False,
) -> str:
    """Create a zip file from all the files under 'base_dir'.

    The output zip file will be named 'base_name' + ".zip".  Uses either the
    "zipfile" Python module (if available) or the InfoZIP "zip" utility
    (if installed and found on the default search path).  If neither tool is
    available, raises DistutilsExecError.  Returns the name of the output zip
    file.
    """
    zip_filename = base_name + ".zip"
    mkpath(os.path.dirname(zip_filename), dry_run=dry_run)

    # If zipfile module is not available, try spawning an external
    # 'zip' command.
    if zipfile is None:
        if verbose:
            zipoptions = "-r"
        else:
            zipoptions = "-rq"

        try:
            spawn(["zip", zipoptions, zip_filename, base_dir], dry_run=dry_run)
        except DistutilsExecError:
            # XXX really should distinguish between "couldn't find
            # external 'zip' command" and "zip failed".
            raise DistutilsExecError(
                f"unable to create zip file '{zip_filename}': "
                "could neither import the 'zipfile' module nor "
                "find a standalone zip utility"
            )

    else:
        log.info("creating '%s' and adding '%s' to it", zip_filename, base_dir)

        if not dry_run:
            try:
                zip = zipfile.ZipFile(
                    zip_filename, "w", compression=zipfile.ZIP_DEFLATED
                )
            except RuntimeError:
                zip = zipfile.ZipFile(zip_filename, "w", compression=zipfile.ZIP_STORED)

            with zip:
                if base_dir != os.curdir:
                    path = os.path.normpath(os.path.join(base_dir, ''))
                    zip.write(path, path)
                    log.info("adding '%s'", path)
                for dirpath, dirnames, filenames in os.walk(base_dir):
                    for name in dirnames:
                        path = os.path.normpath(os.path.join(dirpath, name, ''))
                        zip.write(path, path)
                        log.info("adding '%s'", path)
                    for name in filenames:
                        path = os.path.normpath(os.path.join(dirpath, name))
                        if os.path.isfile(path):
                            zip.write(path, path)
                            log.info("adding '%s'", path)

    return zip_filename


ARCHIVE_FORMATS = {
    'gztar': (make_tarball, [('compress', 'gzip')], "gzip'ed tar-file"),
    'bztar': (make_tarball, [('compress', 'bzip2')], "bzip2'ed tar-file"),
    'xztar': (make_tarball, [('compress', 'xz')], "xz'ed tar-file"),
    'ztar': (make_tarball, [('compress', 'compress')], "compressed tar file"),
    'tar': (make_tarball, [('compress', None)], "uncompressed tar file"),
    'zip': (make_zipfile, [], "ZIP file"),
}


def check_archive_formats(formats):
    """Returns the first format from the 'format' list that is unknown.

    If all formats are known, returns None
    """
    for format in formats:
        if format not in ARCHIVE_FORMATS:
            return format
    return None


@overload
def make_archive(
    base_name: str,
    format: str,
    root_dir: str | os.PathLike[str] | bytes | os.PathLike[bytes] | None = None,
    base_dir: str | None = None,
    verbose: bool = False,
    dry_run: bool = False,
    owner: str | None = None,
    group: str | None = None,
) -> str: ...
@overload
def make_archive(
    base_name: str | os.PathLike[str],
    format: str,
    root_dir: str | os.PathLike[str] | bytes | os.PathLike[bytes],
    base_dir: str | None = None,
    verbose: bool = False,
    dry_run: bool = False,
    owner: str | None = None,
    group: str | None = None,
) -> str: ...
def make_archive(
    base_name: str | os.PathLike[str],
    format: str,
    root_dir: str | os.PathLike[str] | bytes | os.PathLike[bytes] | None = None,
    base_dir: str | None = None,
    verbose: bool = False,
    dry_run: bool = False,
    owner: str | None = None,
    group: str | None = None,
) -> str:
    """Create an archive file (eg. zip or tar).

    'base_name' is the name of the file to create, minus any format-specific
    extension; 'format' is the archive format: one of "zip", "tar", "gztar",
    "bztar", "xztar", or "ztar".

    'root_dir' is a directory that will be the root directory of the
    archive; ie. we typically chdir into 'root_dir' before creating the
    archive.  'base_dir' is the directory where we start archiving from;
    ie. 'base_dir' will be the common prefix of all files and
    directories in the archive.  'root_dir' and 'base_dir' both default
    to the current directory.  Returns the name of the archive file.

    'owner' and 'group' are used when creating a tar archive. By default,
    uses the current owner and group.
    """
    save_cwd = os.getcwd()
    if root_dir is not None:
        log.debug("changing into '%s'", root_dir)
        base_name = os.path.abspath(base_name)
        if not dry_run:
            os.chdir(root_dir)

    if base_dir is None:
        base_dir = os.curdir

    kwargs = {'dry_run': dry_run}

    try:
        format_info = ARCHIVE_FORMATS[format]
    except KeyError:
        raise ValueError(f"unknown archive format '{format}'")

    func = format_info[0]
    kwargs.update(format_info[1])

    if format != 'zip':
        kwargs['owner'] = owner
        kwargs['group'] = group

    try:
        filename = func(base_name, base_dir, **kwargs)
    finally:
        if root_dir is not None:
            log.debug("changing back to '%s'", save_cwd)
            os.chdir(save_cwd)

    return filename
