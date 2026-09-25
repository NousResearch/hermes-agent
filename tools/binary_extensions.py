"""Binary file extensions to skip for text-based operations (ported from
free-code src/constants/files.ts)."""

# Images, video, audio, archives, executables, documents (.pdf deliberately
# excluded — text-based, agents may want to inspect), fonts, bytecode/VM,
# databases, design/3D, Flash, lock/profiling data.
BINARY_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".tiff", ".tif",
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".flv", ".m4v", ".mpeg", ".mpg",
    ".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a", ".wma", ".aiff", ".opus",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar", ".xz", ".z", ".tgz", ".iso",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".a", ".obj", ".lib", ".app", ".msi", ".deb", ".rpm",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".odt", ".ods", ".odp",
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    ".pyc", ".pyo", ".class", ".jar", ".war", ".ear", ".node", ".wasm", ".rlib",
    ".sqlite", ".sqlite3", ".db", ".mdb", ".idx",
    ".psd", ".ai", ".eps", ".sketch", ".fig", ".xd", ".blend", ".3ds", ".max",
    ".swf", ".fla", ".lockb", ".dat", ".data",
})

# Container documents (OOXML/ODF/EPUB zips, OLE, RTF) a plain-text write can
# NEVER produce validly: read_file auto-extracts them, so writing the text back
# via write_file/patch silently destroys the document. PDF is deliberately
# absent — raw PDF syntax is text-authorable, so only overwrites are dangerous
# (the write guard handles that via is_pdf_path).
OPAQUE_DOCUMENT_EXTENSIONS = frozenset({
    ".doc", ".docx", ".docm", ".xls", ".xlsx", ".xlsm", ".xlsb",
    ".ppt", ".pps", ".pptx", ".pptm", ".ppsx", ".ppsm",
    ".odt", ".ods", ".odp", ".rtf", ".epub",
})

# Extensions shared by a container document and a plain-text format. ``.pot``
# is a legacy PowerPoint template (OLE compound file) but far more often the
# gettext PO template every i18n workflow writes as text (#92131). The suffix
# alone cannot tell them apart, so the write guard sniffs the existing file's
# leading bytes; a new or text ``.pot`` is a translation template.
AMBIGUOUS_DOCUMENT_EXTENSIONS = frozenset({".pot"})

# Signature every OLE compound document (legacy .doc/.xls/.ppt/.pot) starts with.
OLE_COMPOUND_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


# SQLite journal sidecars (``x.db-wal``, ``x.sqlite3-shm``, ``x.db-journal``)
# hang their marker off the database's own extension, so the final ``.suffix``
# is ".db-wal" — never in any extension set — and both the read guard and the
# write guard would treat the raw page bytes as text.
_SQLITE_SIDECAR_MARKERS = ("-wal", "-shm", "-journal")

# Only these suffixes take a sidecar marker; ``report.docx-wal`` is not a
# document and must not be treated as one.
_SQLITE_EXTENSIONS = frozenset({".db", ".sqlite", ".sqlite3"})
assert _SQLITE_EXTENSIONS <= BINARY_EXTENSIONS


def _lower_suffix(path: str) -> str:
    """Lower-cased final ``.suffix`` of ``path`` ("" when there is no dot)."""
    dot = path.rfind(".")
    return "" if dot == -1 else path[dot:].lower()


def _strip_sidecar_marker(suffix: str) -> str | None:
    """Return the database suffix a lower-cased sidecar suffix hangs off
    (``.db-wal`` -> ``.db``), or None when ``suffix`` is not a SQLite sidecar."""
    for marker in _SQLITE_SIDECAR_MARKERS:
        if suffix.endswith(marker):
            base = suffix[: -len(marker)]
            if base in _SQLITE_EXTENSIONS:
                return base
    return None


def _has_extension_in(path: str, extensions: frozenset) -> bool:
    """Case-insensitive check on the final ``.suffix``; pure string, no I/O.
    A SQLite sidecar counts as its database's extension (only SQLite suffixes
    are stripped, so ``x.docx-wal`` stays unrecognised)."""
    suffix = _lower_suffix(path)
    return (_strip_sidecar_marker(suffix) or suffix) in extensions


def is_sqlite_sidecar(path: str) -> bool:
    """True for ``x.db-wal`` / ``x.sqlite3-shm`` / ``x.db-journal`` paths.
    Pure string, no I/O — a sidecar path is never a legitimate text target
    even when no sidecar exists on disk (a checkpointed db has none)."""
    return _strip_sidecar_marker(_lower_suffix(path)) is not None


def has_binary_extension(path: str) -> bool:
    return _has_extension_in(path, BINARY_EXTENSIONS)


def has_opaque_document_extension(path: str) -> bool:
    return _has_extension_in(path, OPAQUE_DOCUMENT_EXTENSIONS)


def has_ambiguous_document_extension(path: str) -> bool:
    return _has_extension_in(path, AMBIGUOUS_DOCUMENT_EXTENSIONS)


def is_pdf_path(path: str) -> bool:
    return path.lower().endswith(".pdf")
