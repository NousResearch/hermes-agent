import sys
import unicodedata
from configparser import RawConfigParser

from .compat import py39
from .warnings import SetuptoolsDeprecationWarning


# HFS Plus uses decomposed UTF-8
def decompose(path):
    if isinstance(path, str):
        return unicodedata.normalize('NFD', path)
    try:
        path = path.decode('utf-8')
        path = unicodedata.normalize('NFD', path)
        path = path.encode('utf-8')
    except UnicodeError:
        pass  # Not UTF-8
    return path


def filesys_decode(path):
    """
    Ensure that the given path is decoded,
    ``None`` when no expected encoding works
    """

    if isinstance(path, str):
        return path

    fs_enc = sys.getfilesystemencoding() or 'utf-8'
    candidates = fs_enc, 'utf-8'

    for enc in candidates:
        try:
            return path.decode(enc)
        except UnicodeDecodeError:
            continue

    return None


def try_encode(string, enc):
    "turn unicode encoding into a functional routine"
    try:
        return string.encode(enc)
    except UnicodeEncodeError:
        return None


def _read_utf8_with_fallback(file: str, fallback_encoding=py39.LOCALE_ENCODING) -> str:
    """
    First try to read the file with UTF-8, if there is an error fallback to a
    different encoding ("locale" by default). Returns the content of the file.
    Also useful when reading files that might have been produced by an older version of
    setuptools.
    """
    try:
        with open(file, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:  # pragma: no cover
        _Utf8EncodingNeeded.emit(file=file, fallback_encoding=fallback_encoding)
        with open(file, "r", encoding=fallback_encoding) as f:
            return f.read()


def _cfg_read_utf8_with_fallback(
    cfg: RawConfigParser, file: str, fallback_encoding=py39.LOCALE_ENCODING
) -> None:
    """Same idea as :func:`_read_utf8_with_fallback`, but for the
    :meth:`RawConfigParser.read` method.

    This method may call ``cfg.clear()``.
    """
    try:
        cfg.read(file, encoding="utf-8")
    except UnicodeDecodeError:  # pragma: no cover
        _Utf8EncodingNeeded.emit(file=file, fallback_encoding=fallback_encoding)
        cfg.clear()
        cfg.read(file, encoding=fallback_encoding)


class _Utf8EncodingNeeded(SetuptoolsDeprecationWarning):
    _SUMMARY = """
    `encoding="utf-8"` fails with {file!r}, trying `encoding={fallback_encoding!r}`.
    """

    _DETAILS = """
    Fallback behavior for UTF-8 is considered **deprecated** and future versions of
    `setuptools` may not implement it.

    Please encode {file!r} with "utf-8" to ensure future builds will succeed.

    If this file was produced by `setuptools` itself, cleaning up the cached files
    and re-building/re-installing the package with a newer version of `setuptools`
    (e.g. by updating `build-system.requires` in its `pyproject.toml`)
    might solve the problem.
    """
    # TODO: Add a deadline?
    #       Will we be able to remove this?
    #       The question comes to mind mainly because of sdists that have been produced
    #       by old versions of setuptools and published to PyPI...
