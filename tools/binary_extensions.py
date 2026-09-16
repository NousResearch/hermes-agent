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
    ".ppt", ".pps", ".pot", ".pptx", ".pptm", ".ppsx", ".ppsm",
    ".odt", ".ods", ".odp", ".rtf", ".epub",
})


def _has_extension_in(path: str, extensions: frozenset) -> bool:
    """Case-insensitive check on the final ``.suffix``; pure string, no I/O."""
    dot = path.rfind(".")
    return dot != -1 and path[dot:].lower() in extensions


def has_binary_extension(path: str) -> bool:
    return _has_extension_in(path, BINARY_EXTENSIONS)


def has_opaque_document_extension(path: str) -> bool:
    return _has_extension_in(path, OPAQUE_DOCUMENT_EXTENSIONS)


def is_pdf_path(path: str) -> bool:
    return path.lower().endswith(".pdf")
