"""PyPI and direct package downloading."""

from __future__ import annotations

import base64
import configparser
import hashlib
import html
import http.client
import io
import itertools
import os
import re
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from fnmatch import translate
from functools import wraps
from typing import NamedTuple

from more_itertools import unique_everseen

import setuptools
from pkg_resources import (
    BINARY_DIST,
    CHECKOUT_DIST,
    DEVELOP_DIST,
    EGG_DIST,
    SOURCE_DIST,
    Distribution,
    Environment,
    Requirement,
    find_distributions,
    normalize_path,
    parse_version,
    safe_name,
    safe_version,
    to_filename,
)
from setuptools.wheel import Wheel

from .unicode_utils import _cfg_read_utf8_with_fallback, _read_utf8_with_fallback

from distutils import log
from distutils.errors import DistutilsError

EGG_FRAGMENT = re.compile(r'^egg=([-A-Za-z0-9_.+!]+)$')
HREF = re.compile(r"""href\s*=\s*['"]?([^'"> ]+)""", re.I)
PYPI_MD5 = re.compile(
    r'<a href="([^"#]+)">([^<]+)</a>\n\s+\(<a (?:title="MD5 hash"\n\s+)'
    r'href="[^?]+\?:action=show_md5&amp;digest=([0-9a-f]{32})">md5</a>\)'
)
URL_SCHEME = re.compile('([-+.a-z0-9]{2,}):', re.I).match
EXTENSIONS = ".tar.gz .tar.bz2 .tar .zip .tgz".split()

__all__ = [
    'PackageIndex',
    'distros_for_url',
    'parse_bdist_wininst',
    'interpret_distro_name',
]

_SOCKET_TIMEOUT = 15

user_agent = f"setuptools/{setuptools.__version__} Python-urllib/{sys.version_info.major}.{sys.version_info.minor}"


def parse_requirement_arg(spec):
    try:
        return Requirement.parse(spec)
    except ValueError as e:
        raise DistutilsError(
            f"Not a URL, existing file, or requirement spec: {spec!r}"
        ) from e


def parse_bdist_wininst(name):
    """Return (base,pyversion) or (None,None) for possible .exe name"""

    lower = name.lower()
    base, py_ver, plat = None, None, None

    if lower.endswith('.exe'):
        if lower.endswith('.win32.exe'):
            base = name[:-10]
            plat = 'win32'
        elif lower.startswith('.win32-py', -16):
            py_ver = name[-7:-4]
            base = name[:-16]
            plat = 'win32'
        elif lower.endswith('.win-amd64.exe'):
            base = name[:-14]
            plat = 'win-amd64'
        elif lower.startswith('.win-amd64-py', -20):
            py_ver = name[-7:-4]
            base = name[:-20]
            plat = 'win-amd64'
    return base, py_ver, plat


def egg_info_for_url(url):
    parts = urllib.parse.urlparse(url)
    _scheme, server, path, _parameters, _query, fragment = parts
    base = urllib.parse.unquote(path.split('/')[-1])
    if server == 'sourceforge.net' and base == 'download':  # XXX Yuck
        base = urllib.parse.unquote(path.split('/')[-2])
    if '#' in base:
        base, fragment = base.split('#', 1)
    return base, fragment


def distros_for_url(url, metadata=None):
    """Yield egg or source distribution objects that might be found at a URL"""
    base, fragment = egg_info_for_url(url)
    yield from distros_for_location(url, base, metadata)
    if fragment:
        match = EGG_FRAGMENT.match(fragment)
        if match:
            yield from interpret_distro_name(
                url, match.group(1), metadata, precedence=CHECKOUT_DIST
            )


def distros_for_location(location, basename, metadata=None):
    """Yield egg or source distribution objects based on basename"""
    if basename.endswith('.egg.zip'):
        basename = basename[:-4]  # strip the .zip
    if basename.endswith('.egg') and '-' in basename:
        # only one, unambiguous interpretation
        return [Distribution.from_location(location, basename, metadata)]
    if basename.endswith('.whl') and '-' in basename:
        wheel = Wheel(basename)
        if not wheel.is_compatible():
            return []
        return [
            Distribution(
                location=location,
                project_name=wheel.project_name,
                version=wheel.version,
                # Increase priority over eggs.
                precedence=EGG_DIST + 1,
            )
        ]
    if basename.endswith('.exe'):
        win_base, py_ver, platform = parse_bdist_wininst(basename)
        if win_base is not None:
            return interpret_distro_name(
                location, win_base, metadata, py_ver, BINARY_DIST, platform
            )
    # Try source distro extensions (.zip, .tgz, etc.)
    #
    for ext in EXTENSIONS:
        if basename.endswith(ext):
            basename = basename[: -len(ext)]
            return interpret_distro_name(location, basename, metadata)
    return []  # no extension matched


def distros_for_filename(filename, metadata=None):
    """Yield possible egg or source distribution objects based on a filename"""
    return distros_for_location(
        normalize_path(filename), os.path.basename(filename), metadata
    )


def interpret_distro_name(
    location, basename, metadata, py_version=None, precedence=SOURCE_DIST, platform=None
):
    """Generate the interpretation of a source distro name

    Note: if `location` is a filesystem filename, you should call
    ``pkg_resources.normalize_path()`` on it before passing it to this
    routine!
    """

    parts = basename.split('-')
    if not py_version and any(re.match(r'py\d\.\d$', p) for p in parts[2:]):
        # it is a bdist_dumb, not an sdist -- bail out
        return

    # find the pivot (p) that splits the name from the version.
    # infer the version as the first item that has a digit.
    for p in range(len(parts)):
        if parts[p][:1].isdigit():
            break
    else:
        p = len(parts)

    yield Distribution(
        location,
        metadata,
        '-'.join(parts[:p]),
        '-'.join(parts[p:]),
        py_version=py_version,
        precedence=precedence,
        platform=platform,
    )


def unique_values(func):
    """
    Wrap a function returning an iterable such that the resulting iterable
    only ever yields unique items.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        return unique_everseen(func(*args, **kwargs))

    return wrapper


REL = re.compile(r"""<([^>]*\srel\s{0,10}=\s{0,10}['"]?([^'" >]+)[^>]*)>""", re.I)
"""
Regex for an HTML tag with 'rel="val"' attributes.
"""


@unique_values
def find_external_links(url, page):
    """Find rel="homepage" and rel="download" links in `page`, yielding URLs"""

    for match in REL.finditer(page):
        tag, rel = match.groups()
        rels = set(map(str.strip, rel.lower().split(',')))
        if 'homepage' in rels or 'download' in rels:
            for match in HREF.finditer(tag):
                yield urllib.parse.urljoin(url, htmldecode(match.group(1)))

    for tag in ("<th>Home Page", "<th>Download URL"):
        pos = page.find(tag)
        if pos != -1:
            match = HREF.search(page, pos)
            if match:
                yield urllib.parse.urljoin(url, htmldecode(match.group(1)))


class ContentChecker:
    """
    A null content checker that defines the interface for checking content
    """

    def feed(self, block):
        """
        Feed a block of data to the hash.
        """
        return

    def is_valid(self):
        """
        Check the hash. Return False if validation fails.
        """
        return True

    def report(self, reporter, template):
        """
        Call reporter with information about the checker (hash name)
        substituted into the template.
        """
        return


class HashChecker(ContentChecker):
    pattern = re.compile(
        r'(?P<hash_name>sha1|sha224|sha384|sha256|sha512|md5)='
        r'(?P<expected>[a-f0-9]+)'
    )

    def __init__(self, hash_name, expected) -> None:
        self.hash_name = hash_name
        self.hash = hashlib.new(hash_name)
        self.expected = expected

    @classmethod
    def from_url(cls, url):
        "Construct a (possibly null) ContentChecker from a URL"
        fragment = urllib.parse.urlparse(url)[-1]
        if not fragment:
            return ContentChecker()
        match = cls.pattern.search(fragment)
        if not match:
            return ContentChecker()
        return cls(**match.groupdict())

    def feed(self, block):
        self.hash.update(block)

    def is_valid(self):
        return self.hash.hexdigest() == self.expected

    def report(self, reporter, template):
        msg = template % self.hash_name
        return reporter(msg)


class PackageIndex(Environment):
    """A distribution index that scans web pages for download URLs"""

    def __init__(
        self,
        index_url: str = "https://pypi.org/simple/",
        hosts=('*',),
        ca_bundle=None,
        verify_ssl: bool = True,
        *args,
        **kw,
    ) -> None:
        super().__init__(*args, **kw)
        self.index_url = index_url + "/"[: not index_url.endswith('/')]
        self.scanned_urls: dict = {}
        self.fetched_urls: dict = {}
        self.package_pages: dict = {}
        self.allows = re.compile('|'.join(map(translate, hosts))).match
        self.to_scan: list = []
        self.opener = urllib.request.urlopen

    def add(self, dist):
        # ignore invalid versions
        try:
            parse_version(dist.version)
        except Exception:
            return None
        return super().add(dist)

    # FIXME: 'PackageIndex.process_url' is too complex (14)
    def process_url(self, url, retrieve: bool = False) -> None:  # noqa: C901
        """Evaluate a URL as a possible download, and maybe retrieve it"""
        if url in self.scanned_urls and not retrieve:
            return
        self.scanned_urls[url] = True
        if not URL_SCHEME(url):
            self.process_filename(url)
            return
        else:
            dists = list(distros_for_url(url))
            if dists:
                if not self.url_ok(url):
                    return
                self.debug("Found link: %s", url)

        if dists or not retrieve or url in self.fetched_urls:
            list(map(self.add, dists))
            return  # don't need the actual page

        if not self.url_ok(url):
            self.fetched_urls[url] = True
            return

        self.info("Reading %s", url)
        self.fetched_urls[url] = True  # prevent multiple fetch attempts
        tmpl = "Download error on %s: %%s -- Some packages may not be found!"
        f = self.open_url(url, tmpl % url)
        if f is None:
            return
        if isinstance(f, urllib.error.HTTPError) and f.code == 401:
            self.info(f"Authentication error: {f.msg}")
        self.fetched_urls[f.url] = True
        if 'html' not in f.headers.get('content-type', '').lower():
            f.close()  # not html, we can't process it
            return

        base = f.url  # handle redirects
        page = f.read()
        if not isinstance(page, str):
            # In Python 3 and got bytes but want str.
            if isinstance(f, urllib.error.HTTPError):
                # Errors have no charset, assume latin1:
                charset = 'latin-1'
            else:
                charset = f.headers.get_param('charset') or 'latin-1'
            page = page.decode(charset, "ignore")
        f.close()
        for match in HREF.finditer(page):
            link = urllib.parse.urljoin(base, htmldecode(match.group(1)))
            self.process_url(link)
        if url.startswith(self.index_url) and getattr(f, 'code', None) != 404:
            page = self.process_index(url, page)

    def process_filename(self, fn, nested: bool = False) -> None:
        # process filenames or directories
        if not os.path.exists(fn):
            self.warn("Not found: %s", fn)
            return

        if os.path.isdir(fn) and not nested:
            path = os.path.realpath(fn)
            for item in os.listdir(path):
                self.process_filename(os.path.join(path, item), True)

        dists = distros_for_filename(fn)
        if dists:
            self.debug("Found: %s", fn)
            list(map(self.add, dists))

    def url_ok(self, url, fatal: bool = False) -> bool:
        s = URL_SCHEME(url)
        is_file = s and s.group(1).lower() == 'file'
        if is_file or self.allows(urllib.parse.urlparse(url)[1]):
            return True
        msg = (
            "\nNote: Bypassing %s (disallowed host; see "
            "https://setuptools.pypa.io/en/latest/deprecated/"
            "easy_install.html#restricting-downloads-with-allow-hosts for details).\n"
        )
        if fatal:
            raise DistutilsError(msg % url)
        else:
            self.warn(msg, url)
            return False

    def scan_egg_links(self, search_path) -> None:
        dirs = filter(os.path.isdir, search_path)
        egg_links = (
            (path, entry)
            for path in dirs
            for entry in os.listdir(path)
            if entry.endswith('.egg-link')
        )
        list(itertools.starmap(self.scan_egg_link, egg_links))

    def scan_egg_link(self, path, entry) -> None:
        content = _read_utf8_with_fallback(os.path.join(path, entry))
        # filter non-empty lines
        lines = list(filter(None, map(str.strip, content.splitlines())))

        if len(lines) != 2:
            # format is not recognized; punt
            return

        egg_path, _setup_path = lines

        for dist in find_distributions(os.path.join(path, egg_path)):
            dist.location = os.path.join(path, *lines)
            dist.precedence = SOURCE_DIST
            self.add(dist)

    def _scan(self, link):
        # Process a URL to see if it's for a package page
        NO_MATCH_SENTINEL = None, None
        if not link.startswith(self.index_url):
            return NO_MATCH_SENTINEL

        parts = list(map(urllib.parse.unquote, link[len(self.index_url) :].split('/')))
        if len(parts) != 2 or '#' in parts[1]:
            return NO_MATCH_SENTINEL

        # it's a package page, sanitize and index it
        pkg = safe_name(parts[0])
        ver = safe_version(parts[1])
        self.package_pages.setdefault(pkg.lower(), {})[link] = True
        return to_filename(pkg), to_filename(ver)

    def process_index(self, url, page):
        """Process the contents of a PyPI page"""

        # process an index page into the package-page index
        for match in HREF.finditer(page):
            try:
                self._scan(urllib.parse.urljoin(url, htmldecode(match.group(1))))
            except ValueError:
                pass

        pkg, ver = self._scan(url)  # ensure this page is in the page index
        if not pkg:
            return ""  # no sense double-scanning non-package pages

        # process individual package page
        for new_url in find_external_links(url, page):
            # Process the found URL
            base, frag = egg_info_for_url(new_url)
            if base.endswith('.py') and not frag:
                if ver:
                    new_url += f'#egg={pkg}-{ver}'
                else:
                    self.need_version_info(url)
            self.scan_url(new_url)

        return PYPI_MD5.sub(
            lambda m: '<a href="{}#md5={}">{}</a>'.format(*m.group(1, 3, 2)), page
        )

    def need_version_info(self, url) -> None:
        self.scan_all(
            "Page at %s links to .py file(s) without version info; an index "
            "scan is required.",
            url,
        )

    def scan_all(self, msg=None, *args) -> None:
        if self.index_url not in self.fetched_urls:
            if msg:
                self.warn(msg, *args)
            self.info("Scanning index of all packages (this may take a while)")
        self.scan_url(self.index_url)

    def find_packages(self, requirement) -> None:
        self.scan_url(self.index_url + requirement.unsafe_name + '/')

        if not self.package_pages.get(requirement.key):
            # Fall back to safe version of the name
            self.scan_url(self.index_url + requirement.project_name + '/')

        if not self.package_pages.get(requirement.key):
            # We couldn't find the target package, so search the index page too
            self.not_found_in_index(requirement)

        for url in list(self.package_pages.get(requirement.key, ())):
            # scan each page that might be related to the desired package
            self.scan_url(url)

    def obtain(self, requirement, installer=None):
        self.prescan()
        self.find_packages(requirement)
        for dist in self[requirement.key]:
            if dist in requirement:
                return dist
            self.debug("%s does not match %s", requirement, dist)
        return super().obtain(requirement, installer)

    def check_hash(self, checker, filename, tfp) -> None:
        """
        checker is a ContentChecker
        """
        checker.report(self.debug, f"Validating %s checksum for {filename}")
        if not checker.is_valid():
            tfp.close()
            os.unlink(filename)
            raise DistutilsError(
                f"{checker.hash.name} validation failed for {os.path.basename(filename)}; "
                "possible download problem?"
            )

    def add_find_links(self, urls) -> None:
        """Add `urls` to the list that will be prescanned for searches"""
        for url in urls:
            if (
                self.to_scan is None  # if we have already "gone online"
                or not URL_SCHEME(url)  # or it's a local file/directory
                or url.startswith('file:')
                or list(distros_for_url(url))  # or a direct package link
            ):
                # then go ahead and process it now
                self.scan_url(url)
            else:
                # otherwise, defer retrieval till later
                self.to_scan.append(url)

    def prescan(self):
        """Scan urls scheduled for prescanning (e.g. --find-links)"""
        if self.to_scan:
            list(map(self.scan_url, self.to_scan))
        self.to_scan = None  # from now on, go ahead and process immediately

    def not_found_in_index(self, requirement) -> None:
        if self[requirement.key]:  # we've seen at least one distro
            meth, msg = self.info, "Couldn't retrieve index page for %r"
        else:  # no distros seen for this name, might be misspelled
            meth, msg = self.warn, "Couldn't find index page for %r (maybe misspelled?)"
        meth(msg, requirement.unsafe_name)
        self.scan_all()

    def download(self, spec, tmpdir):
        """Locate and/or download `spec` to `tmpdir`, returning a local path

        `spec` may be a ``Requirement`` object, or a string containing a URL,
        an existing local filename, or a project/version requirement spec
        (i.e. the string form of a ``Requirement`` object).  If it is the URL
        of a .py file with an unambiguous ``#egg=name-version`` tag (i.e., one
        that escapes ``-`` as ``_`` throughout), a trivial ``setup.py`` is
        automatically created alongside the downloaded file.

        If `spec` is a ``Requirement`` object or a string containing a
        project/version requirement spec, this method returns the location of
        a matching distribution (possibly after downloading it to `tmpdir`).
        If `spec` is a locally existing file or directory name, it is simply
        returned unchanged.  If `spec` is a URL, it is downloaded to a subpath
        of `tmpdir`, and the local filename is returned.  Various errors may be
        raised if a problem occurs during downloading.
        """
        if not isinstance(spec, Requirement):
            scheme = URL_SCHEME(spec)
            if scheme:
                # It's a url, download it to tmpdir
                found = self._download_url(spec, tmpdir)
                base, fragment = egg_info_for_url(spec)
                if base.endswith('.py'):
                    found = self.gen_setup(found, fragment, tmpdir)
                return found
            elif os.path.exists(spec):
                # Existing file or directory, just return it
                return spec
            else:
                spec = parse_requirement_arg(spec)
        return getattr(self.fetch_distribution(spec, tmpdir), 'location', None)

    def fetch_distribution(  # noqa: C901  # is too complex (14)  # FIXME
        self,
        requirement,
        tmpdir,
        force_scan: bool = False,
        source: bool = False,
        develop_ok: bool = False,
        local_index=None,
    ) -> Distribution | None:
        """Obtain a distribution suitable for fulfilling `requirement`

        `requirement` must be a ``pkg_resources.Requirement`` instance.
        If necessary, or if the `force_scan` flag is set, the requirement is
        searched for in the (online) package index as well as the locally
        installed packages.  If a distribution matching `requirement` is found,
        the returned distribution's ``location`` is the value you would have
        gotten from calling the ``download()`` method with the matching
        distribution's URL or filename.  If no matching distribution is found,
        ``None`` is returned.

        If the `source` flag is set, only source distributions and source
        checkout links will be considered.  Unless the `develop_ok` flag is
        set, development and system eggs (i.e., those using the ``.egg-info``
        format) will be ignored.
        """
        # process a Requirement
        self.info("Searching for %s", requirement)
        skipped = set()
        dist = None

        def find(req, env: Environment | None = None):
            if env is None:
                env = self
            # Find a matching distribution; may be called more than once

            for dist in env[req.key]:
                if dist.precedence == DEVELOP_DIST and not develop_ok:
                    if dist not in skipped:
                        self.warn(
                            "Skipping development or system egg: %s",
                            dist,
                        )
                        skipped.add(dist)
                    continue

                test = dist in req and (dist.precedence <= SOURCE_DIST or not source)
                if test:
                    loc = self.download(dist.location, tmpdir)
                    dist.download_location = loc
                    if os.path.exists(dist.download_location):
                        return dist

            return None

        if force_scan:
            self.prescan()
            self.find_packages(requirement)
            dist = find(requirement)

        if not dist and local_index is not None:
            dist = find(requirement, local_index)

        if dist is None:
            if self.to_scan is not None:
                self.prescan()
            dist = find(requirement)

        if dist is None and not force_scan:
            self.find_packages(requirement)
            dist = find(requirement)

        if dist is None:
            self.warn(
                "No local packages or working download links found for %s%s",
                (source and "a source distribution of " or ""),
                requirement,
            )
            return None
        else:
            self.info("Best match: %s", dist)
            return dist.clone(location=dist.download_location)

    def fetch(
        self, requirement, tmpdir, force_scan: bool = False, source: bool = False
    ) -> str | None:
        """Obtain a file suitable for fulfilling `requirement`

        DEPRECATED; use the ``fetch_distribution()`` method now instead.  For
        backward compatibility, this routine is identical but returns the
        ``location`` of the downloaded distribution instead of a distribution
        object.
        """
        dist = self.fetch_distribution(requirement, tmpdir, force_scan, source)
        if dist is not None:
            return dist.location
        return None

    def gen_setup(self, filename, fragment, tmpdir):
        match = EGG_FRAGMENT.match(fragment)
        dists = (
            match
            and [
                d
                for d in interpret_distro_name(filename, match.group(1), None)
                if d.version
            ]
            or []
        )

        if len(dists) == 1:  # unambiguous ``#egg`` fragment
            basename = os.path.basename(filename)

            # Make sure the file has been downloaded to the temp dir.
            if os.path.dirname(filename) != tmpdir:
                dst = os.path.join(tmpdir, basename)
                if not (os.path.exists(dst) and os.path.samefile(filename, dst)):
                    shutil.copy2(filename, dst)
                    filename = dst

            with open(os.path.join(tmpdir, 'setup.py'), 'w', encoding="utf-8") as file:
                file.write(
                    "from setuptools import setup\n"
                    f"setup(name={dists[0].project_name!r}, version={dists[0].version!r}, py_modules=[{os.path.splitext(basename)[0]!r}])\n"
                )
            return filename

        elif match:
            raise DistutilsError(
                f"Can't unambiguously interpret project/version identifier {fragment!r}; "
                "any dashes in the name or version should be escaped using "
                f"underscores. {dists!r}"
            )
        else:
            raise DistutilsError(
                "Can't process plain .py files without an '#egg=name-version'"
                " suffix to enable automatic setup script generation."
            )

    dl_blocksize = 8192

    def _download_to(self, url, filename):
        self.info("Downloading %s", url)
        # Download the file
        fp = None
        try:
            checker = HashChecker.from_url(url)
            fp = self.open_url(url)
            if isinstance(fp, urllib.error.HTTPError):
                raise DistutilsError(f"Can't download {url}: {fp.code} {fp.msg}")
            headers = fp.info()
            blocknum = 0
            bs = self.dl_blocksize
            size = -1
            if "content-length" in headers:
                # Some servers return multiple Content-Length headers :(
                sizes = headers.get_all('Content-Length')
                size = max(map(int, sizes))
                self.reporthook(url, filename, blocknum, bs, size)
            with open(filename, 'wb') as tfp:
                while True:
                    block = fp.read(bs)
                    if block:
                        checker.feed(block)
                        tfp.write(block)
                        blocknum += 1
                        self.reporthook(url, filename, blocknum, bs, size)
                    else:
                        break
                self.check_hash(checker, filename, tfp)
            return headers
        finally:
            if fp:
                fp.close()

    def reporthook(self, url, filename, blocknum, blksize, size) -> None:
        pass  # no-op

    # FIXME:
    def open_url(self, url, warning=None):  # noqa: C901  # is too complex (12)
        if url.startswith('file:'):
            return local_open(url)
        try:
            return open_with_auth(url, self.opener)
        except (ValueError, http.client.InvalidURL) as v:
            msg = ' '.join([str(arg) for arg in v.args])
            if warning:
                self.warn(warning, msg)
            else:
                raise DistutilsError(f'{url} {msg}') from v
        except urllib.error.HTTPError as v:
            return v
        except urllib.error.URLError as v:
            if warning:
                self.warn(warning, v.reason)
            else:
                raise DistutilsError(f"Download error for {url}: {v.reason}") from v
        except http.client.BadStatusLine as v:
            if warning:
                self.warn(warning, v.line)
            else:
                raise DistutilsError(
                    f'{url} returned a bad status line. The server might be '
                    f'down, {v.line}'
                ) from v
        except (http.client.HTTPException, OSError) as v:
            if warning:
                self.warn(warning, v)
            else:
                raise DistutilsError(f"Download error for {url}: {v}") from v

    @staticmethod
    def _sanitize(name):
        r"""
        Replace unsafe path directives with underscores.

        >>> san = PackageIndex._sanitize
        >>> san('/home/user/.ssh/authorized_keys')
        '_home_user_.ssh_authorized_keys'
        >>> san('..\\foo\\bing')
        '__foo_bing'
        >>> san('D:bar')
        'D_bar'
        >>> san('C:\\bar')
        'C__bar'
        >>> san('foo..bar')
        'foo..bar'
        >>> san('D:../foo')
        'D___foo'
        """
        pattern = '|'.join((
            # drive letters
            r':',
            # path separators
            r'[/\\]',
            # parent dirs
            r'(?:(?<=([/\\]|:))\.\.(?=[/\\]|$))|(?:^\.\.(?=[/\\]|$))',
        ))
        return re.sub(pattern, r'_', name)

    @classmethod
    def _resolve_download_filename(cls, url, tmpdir):
        """
        >>> import pathlib
        >>> du = PackageIndex._resolve_download_filename
        >>> root = getfixture('tmp_path')
        >>> url = 'https://files.pythonhosted.org/packages/a9/5a/0db.../setuptools-78.1.0.tar.gz'
        >>> str(pathlib.Path(du(url, root)).relative_to(root))
        'setuptools-78.1.0.tar.gz'
        """
        name, _fragment = egg_info_for_url(url)
        name = cls._sanitize(
            name
            or
            # default if URL has no path contents
            '__downloaded__'
        )

        # strip any extra .zip before download
        name = re.sub(r'\.egg\.zip$', '.egg', name)

        return os.path.join(tmpdir, name)

    def _download_url(self, url, tmpdir):
        """
        Determine the download filename.
        """
        filename = self._resolve_download_filename(url, tmpdir)
        return self._download_vcs(url, filename) or self._download_other(url, filename)

    @staticmethod
    def _resolve_vcs(url):
        """
        >>> rvcs = PackageIndex._resolve_vcs
        >>> rvcs('git+http://foo/bar')
        'git'
        >>> rvcs('hg+https://foo/bar')
        'hg'
        >>> rvcs('git:myhost')
        'git'
        >>> rvcs('hg:myhost')
        >>> rvcs('http://foo/bar')
        """
        scheme = urllib.parse.urlsplit(url).scheme
        pre, sep, _post = scheme.partition('+')
        # svn and git have their own protocol; hg does not
        allowed = set(['svn', 'git'] + ['hg'] * bool(sep))
        return next(iter({pre} & allowed), None)

    def _download_vcs(self, url, spec_filename):
        vcs = self._resolve_vcs(url)
        if not vcs:
            return None
        if vcs == 'svn':
            raise DistutilsError(
                f"Invalid config, SVN download is not supported: {url}"
            )

        filename, _, _ = spec_filename.partition('#')
        url, rev = self._vcs_split_rev_from_url(url)

        self.info(f"Doing {vcs} clone from {url} to {filename}")
        subprocess.check_call([vcs, 'clone', '--quiet', url, filename])

        co_commands = dict(
            git=[vcs, '-C', filename, 'checkout', '--quiet', rev],
            hg=[vcs, '--cwd', filename, 'up', '-C', '-r', rev, '-q'],
        )
        if rev is not None:
            self.info(f"Checking out {rev}")
            subprocess.check_call(co_commands[vcs])

        return filename

    def _download_other(self, url, filename):
        scheme = urllib.parse.urlsplit(url).scheme
        if scheme == 'file':  # pragma: no cover
            return urllib.request.url2pathname(urllib.parse.urlparse(url).path)
        # raise error if not allowed
        self.url_ok(url, True)
        return self._attempt_download(url, filename)

    def scan_url(self, url) -> None:
        self.process_url(url, True)

    def _attempt_download(self, url, filename):
        headers = self._download_to(url, filename)
        if 'html' in headers.get('content-type', '').lower():
            return self._invalid_download_html(url, headers, filename)
        else:
            return filename

    def _invalid_download_html(self, url, headers, filename):
        os.unlink(filename)
        raise DistutilsError(f"Unexpected HTML page found at {url}")

    @staticmethod
    def _vcs_split_rev_from_url(url):
        """
        Given a possible VCS URL, return a clean URL and resolved revision if any.

        >>> vsrfu = PackageIndex._vcs_split_rev_from_url
        >>> vsrfu('git+https://github.com/pypa/setuptools@v69.0.0#egg-info=setuptools')
        ('https://github.com/pypa/setuptools', 'v69.0.0')
        >>> vsrfu('git+https://github.com/pypa/setuptools#egg-info=setuptools')
        ('https://github.com/pypa/setuptools', None)
        >>> vsrfu('http://foo/bar')
        ('http://foo/bar', None)
        """
        parts = urllib.parse.urlsplit(url)

        clean_scheme = parts.scheme.split('+', 1)[-1]

        # Some fragment identification fails
        no_fragment_path, _, _ = parts.path.partition('#')

        pre, sep, post = no_fragment_path.rpartition('@')
        clean_path, rev = (pre, post) if sep else (post, None)

        resolved = parts._replace(
            scheme=clean_scheme,
            path=clean_path,
            # discard the fragment
            fragment='',
        ).geturl()

        return resolved, rev

    def debug(self, msg, *args) -> None:
        log.debug(msg, *args)

    def info(self, msg, *args) -> None:
        log.info(msg, *args)

    def warn(self, msg, *args) -> None:
        log.warn(msg, *args)


# This pattern matches a character entity reference (a decimal numeric
# references, a hexadecimal numeric reference, or a named reference).
entity_sub = re.compile(r'&(#(\d+|x[\da-fA-F]+)|[\w.:-]+);?').sub


def decode_entity(match):
    what = match.group(0)
    return html.unescape(what)


def htmldecode(text):
    """
    Decode HTML entities in the given text.

    >>> htmldecode(
    ...     'https://../package_name-0.1.2.tar.gz'
    ...     '?tokena=A&amp;tokenb=B">package_name-0.1.2.tar.gz')
    'https://../package_name-0.1.2.tar.gz?tokena=A&tokenb=B">package_name-0.1.2.tar.gz'
    """
    return entity_sub(decode_entity, text)


def socket_timeout(timeout=15):
    def _socket_timeout(func):
        def _socket_timeout(*args, **kwargs):
            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(timeout)
            try:
                return func(*args, **kwargs)
            finally:
                socket.setdefaulttimeout(old_timeout)

        return _socket_timeout

    return _socket_timeout


def _encode_auth(auth):
    """
    Encode auth from a URL suitable for an HTTP header.
    >>> str(_encode_auth('username%3Apassword'))
    'dXNlcm5hbWU6cGFzc3dvcmQ='

    Long auth strings should not cause a newline to be inserted.
    >>> long_auth = 'username:' + 'password'*10
    >>> chr(10) in str(_encode_auth(long_auth))
    False
    """
    auth_s = urllib.parse.unquote(auth)
    # convert to bytes
    auth_bytes = auth_s.encode()
    encoded_bytes = base64.b64encode(auth_bytes)
    # convert back to a string
    encoded = encoded_bytes.decode()
    # strip the trailing carriage return
    return encoded.replace('\n', '')


class Credential(NamedTuple):
    """
    A username/password pair.

    Displayed separated by `:`.
    >>> str(Credential('username', 'password'))
    'username:password'
    """

    username: str
    password: str

    def __str__(self) -> str:
        return f'{self.username}:{self.password}'


class PyPIConfig(configparser.RawConfigParser):
    def __init__(self):
        """
        Load from ~/.pypirc
        """
        defaults = dict.fromkeys(['username', 'password', 'repository'], '')
        super().__init__(defaults)

        rc = os.path.join(os.path.expanduser('~'), '.pypirc')
        if os.path.exists(rc):
            _cfg_read_utf8_with_fallback(self, rc)

    @property
    def creds_by_repository(self):
        sections_with_repositories = [
            section
            for section in self.sections()
            if self.get(section, 'repository').strip()
        ]

        return dict(map(self._get_repo_cred, sections_with_repositories))

    def _get_repo_cred(self, section):
        repo = self.get(section, 'repository').strip()
        return repo, Credential(
            self.get(section, 'username').strip(),
            self.get(section, 'password').strip(),
        )

    def find_credential(self, url):
        """
        If the URL indicated appears to be a repository defined in this
        config, return the credential for that repository.
        """
        for repository, cred in self.creds_by_repository.items():
            if url.startswith(repository):
                return cred
        return None


def open_with_auth(url, opener=urllib.request.urlopen):
    """Open a urllib2 request, handling HTTP authentication"""

    parsed = urllib.parse.urlparse(url)
    scheme, netloc, path, params, query, frag = parsed

    # Double scheme does not raise on macOS as revealed by a
    # failing test. We would expect "nonnumeric port". Refs #20.
    if netloc.endswith(':'):
        raise http.client.InvalidURL("nonnumeric port: ''")

    if scheme in ('http', 'https'):
        auth, address = _splituser(netloc)
    else:
        auth, address = (None, None)

    if not auth:
        cred = PyPIConfig().find_credential(url)
        if cred:
            auth = str(cred)
            info = cred.username, url
            log.info('Authenticating as %s for %s (from .pypirc)', *info)

    if auth:
        auth = "Basic " + _encode_auth(auth)
        parts = scheme, address, path, params, query, frag
        new_url = urllib.parse.urlunparse(parts)
        request = urllib.request.Request(new_url)
        request.add_header("Authorization", auth)
    else:
        request = urllib.request.Request(url)

    request.add_header('User-Agent', user_agent)
    fp = opener(request)

    if auth:
        # Put authentication info back into request URL if same host,
        # so that links found on the page will work
        s2, h2, path2, param2, query2, frag2 = urllib.parse.urlparse(fp.url)
        if s2 == scheme and h2 == address:
            parts = s2, netloc, path2, param2, query2, frag2
            fp.url = urllib.parse.urlunparse(parts)

    return fp


# copy of urllib.parse._splituser from Python 3.8
# See https://github.com/python/cpython/issues/80072.
def _splituser(host):
    """splituser('user[:passwd]@host[:port]')
    --> 'user[:passwd]', 'host[:port]'."""
    user, delim, host = host.rpartition('@')
    return (user if delim else None), host


# adding a timeout to avoid freezing package_index
open_with_auth = socket_timeout(_SOCKET_TIMEOUT)(open_with_auth)


def fix_sf_url(url):
    return url  # backward compatibility


def local_open(url):
    """Read a local path, with special support for directories"""
    _scheme, _server, path, _param, _query, _frag = urllib.parse.urlparse(url)
    filename = urllib.request.url2pathname(path)
    if os.path.isfile(filename):
        return urllib.request.urlopen(url)
    elif path.endswith('/') and os.path.isdir(filename):
        files = []
        for f in os.listdir(filename):
            filepath = os.path.join(filename, f)
            if f == 'index.html':
                body = _read_utf8_with_fallback(filepath)
                break
            elif os.path.isdir(filepath):
                f += '/'
            files.append(f'<a href="{f}">{f}</a>')
        else:
            tmpl = "<html><head><title>{url}</title></head><body>{files}</body></html>"
            body = tmpl.format(url=url, files='\n'.join(files))
        status, message = 200, "OK"
    else:
        status, message, body = 404, "Path not found", "Not found"

    headers = {'content-type': 'text/html'}
    body_stream = io.StringIO(body)
    return urllib.error.HTTPError(url, status, message, headers, body_stream)
