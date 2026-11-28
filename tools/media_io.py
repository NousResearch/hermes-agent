"""Shared bounded, symlink-safe media file I/O and MIME/magic validation helpers."""
from __future__ import annotations

import base64
import os
import re
import stat
import zlib
from typing import Callable, Optional

from tools.audio_container import sniff_container

MAX_NATIVE_MEDIA_RAW_BYTES = 15 * 1024 * 1024  # ~20 MB base64 inline cap
_MAX_PDF_SCAN_DECOMPRESS_BYTES = 20 * 1024 * 1024
_MAX_PDF_FILTER_CHAIN_DEPTH = 4
_PDF_ENDSTREAM_RE = re.compile(rb"endstream")
_PDF_STREAM_START_RE = re.compile(rb"(?<![A-Za-z0-9_])stream(?:\r\n|\n)")
_PDF_TJ_SPLIT_RE = re.compile(r"\)\s*(?:-?\d+(?:\.\d+)?\s*)?\(")
_PDF_NAME_TOKEN_RE = re.compile(rb"/[A-Za-z0-9#._-]+")
_PDF_NAME_HEX_RE = re.compile(rb"#([0-9A-Fa-f]{2})")
_PDF_FILTER_ARRAY_RE = re.compile(rb"/Filter\s*\[([^\]]*)\]", re.DOTALL)
_PDF_FILTER_SINGLE_RE = re.compile(rb"/Filter\s*/([A-Za-z0-9._-]+)")
_PDF_PREDICTOR_RE = re.compile(rb"/Predictor\s+(\d+)")
_PDF_SUBTYPE_IMAGE_RE = re.compile(rb"/Subtype\s*/Image(?![A-Za-z0-9._-])")
_PDF_STRING_TOKEN_RE = re.compile(
    rb"\(((?:\\.|[^\\)])*)\)|(?<!<)<([0-9A-Fa-f\s]+)>(?!>)",
    re.DOTALL,
)
_PDF_LIT_ESC_RE = re.compile(rb"\\(\r\n|\r|\n|[0-7]{1,3}|.)", re.DOTALL)
_PDF_LIT_SIMPLE_ESCAPES: dict[bytes, bytes] = {
    b"n": b"\n",
    b"r": b"\r",
    b"t": b"\t",
    b"b": b"\b",
    b"f": b"\f",
}
_PDF_IMAGE_ONLY_FILTERS = frozenset({
    b"DCTDecode", b"DCT", b"JPXDecode", b"JBIG2Decode", b"CCITTFaxDecode", b"CCF",
})

_AUDIO_MIME_ALIASES: dict[str, str] = {
    "audio/x-wav": "audio/wav",
    "audio/wave": "audio/wav",
    "audio/x-pn-wav": "audio/wav",
    "audio/x-flac": "audio/flac",
    "audio/x-m4a": "audio/mp4",
    "audio/m4a": "audio/mp4",
    "audio/x-mp4": "audio/mp4",
    "audio/x-aiff": "audio/aiff",
    "audio/x-mpeg": "audio/mpeg",
    "audio/mp3": "audio/mpeg",
    "audio/x-ogg": "audio/ogg",
    "audio/x-aac": "audio/aac",
}

_TEXT_SPOOF_PREFIXES = (
    b"<!doctype",
    b"<html",
    b"<head",
    b"<body",
    b"<script",
    b"<?xml",
    b"<svg",
    b"{",
    b"[",
)


def normalize_audio_mime(mime: str) -> str:
    """Normalize non-canonical ``audio/x-*`` MIME aliases to canonical Gemini/Interactions types."""
    low = (mime or "").split(";", 1)[0].strip().lower()
    return _AUDIO_MIME_ALIASES.get(low, low)


def _open_regular_nofollow(path: str, max_file_bytes: int) -> Optional[int]:
    """Open a regular local file without following symlinks or blocking on FIFOs.

    Uses ``O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_BINARY``. On platforms lacking
    ``O_NOFOLLOW`` (Windows), performs an ``os.lstat`` symlink pre-check and
    verifies inode/device identity after ``os.fstat``.
    """
    if not path or max_file_bytes <= 0:
        return None
    has_nofollow = hasattr(os, "O_NOFOLLOW")
    lst = None
    if not has_nofollow:
        try:
            lst = os.lstat(path)
        except OSError:
            return None
        if stat.S_ISLNK(lst.st_mode) or not stat.S_ISREG(lst.st_mode):
            return None
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_BINARY", 0)
    )
    try:
        fd = os.open(path, flags)
    except OSError:
        return None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size <= 0 or st.st_size > max_file_bytes:
            os.close(fd)
            return None
        if lst is not None:
            if (lst.st_ino and st.st_ino != lst.st_ino) or (lst.st_dev and st.st_dev != lst.st_dev):
                os.close(fd)
                return None
    except OSError:
        try:
            os.close(fd)
        except OSError:
            pass
        return None
    return fd


def peek_file_header_nofollow(
    path: str,
    *,
    max_file_bytes: int = MAX_NATIVE_MEDIA_RAW_BYTES,
    header_bytes: int = 64,
) -> Optional[bytes]:
    """Validate a regular local file via ``O_NOFOLLOW`` + ``O_NONBLOCK`` + ``fstat``
    (``0 < size <= max_file_bytes``) and return only its first *header_bytes* without
    allocating the full file in RAM."""
    if header_bytes <= 0:
        return None
    fd = _open_regular_nofollow(path, max_file_bytes)
    if fd is None:
        return None
    try:
        fp = os.fdopen(fd, "rb")
        fd = -1
        with fp:
            head = fp.read(header_bytes)
        return head if head else None
    except OSError:
        return None
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass


def read_file_bounded_nofollow(path: str, *, max_bytes: int = MAX_NATIVE_MEDIA_RAW_BYTES) -> Optional[bytes]:
    """Read a regular local file without following symlinks and without loading oversized files into RAM.

    Returns ``None`` on symlink (``ELOOP``), non-regular file (FIFO/device/dir), empty/oversized file,
    or OS error. Strictly closes ``fd`` on every error path without double-closing after ``os.fdopen`` handoff.
    """
    fd = _open_regular_nofollow(path, max_bytes)
    if fd is None:
        return None
    try:
        fp = os.fdopen(fd, "rb")
        fd = -1  # ownership transferred to fp; finally will not double-close
        with fp:
            raw = bytearray()
            while len(raw) <= max_bytes:
                chunk = fp.read(min(65536, max_bytes - len(raw) + 1))
                if not chunk:
                    break
                raw.extend(chunk)
        return bytes(raw) if 0 < len(raw) <= max_bytes else None
    except OSError:
        return None
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass


def is_valid_pdf_bytes(data: bytes) -> bool:
    """True when *data* begins with a genuine ``%PDF-`` header."""
    return bool(data) and data[:5] == b"%PDF-"


def _decode_pdf_names(buf: bytes) -> bytes:
    """Decode ISO 32000-1 ``#XX`` hex escapes inside PDF ``/Name`` tokens."""
    if b"#" not in buf:
        return buf

    def _unescape_name(match: re.Match[bytes]) -> bytes:
        token = match.group(0)
        if b"#" not in token:
            return token
        return _PDF_NAME_HEX_RE.sub(lambda hm: bytes([int(hm.group(1), 16)]), token)

    return _PDF_NAME_TOKEN_RE.sub(_unescape_name, buf)


def _unescape_pdf_literal(raw: bytes) -> bytes:
    """Unescape a PDF literal string body per ISO 32000-1 §7.3.4.2."""
    if b"\\" not in raw:
        return raw

    def _sub(match: re.Match[bytes]) -> bytes:
        tok = match.group(1)
        if tok in (b"\r\n", b"\r", b"\n"):
            return b""
        if tok[0] in b"01234567":
            return bytes([int(tok, 8) & 0xFF])
        return _PDF_LIT_SIMPLE_ESCAPES.get(tok, tok)

    return _PDF_LIT_ESC_RE.sub(_sub, raw)


def _decode_pdf_string_bytes(raw: bytes) -> str:
    if raw.startswith(b"\xfe\xff"):
        return raw[2:].decode("utf-16-be", errors="ignore")
    if raw.startswith(b"\xff\xfe"):
        return raw[2:].decode("utf-16-le", errors="ignore")
    return raw.decode("latin-1", errors="ignore")


def _decode_pdf_hex_string(hex_bytes: bytes) -> str:
    cleaned = b"".join(hex_bytes.split())
    if not cleaned:
        return ""
    if len(cleaned) % 2:
        cleaned += b"0"
    try:
        raw = bytes.fromhex(cleaned.decode("ascii"))
    except ValueError:
        return ""
    return _decode_pdf_string_bytes(raw)


def _extract_pdf_strings(buf: bytes) -> list[str]:
    """Extract decoded literal ``(...)`` and hex ``<...>`` PDF strings in document order."""
    out: list[str] = []
    for match in _PDF_STRING_TOKEN_RE.finditer(buf):
        lit, hx = match.group(1), match.group(2)
        if lit is not None:
            text = _decode_pdf_string_bytes(_unescape_pdf_literal(lit))
        else:
            text = _decode_pdf_hex_string(hx)
        if text:
            out.append(text)
    return out


def _text_triggers_redaction(text: str, redactor: Callable[[str], str]) -> bool:
    if not text:
        return False
    if redactor(text) != text:
        return True
    if ")" in text:
        joined = _PDF_TJ_SPLIT_RE.sub("", text)
        if joined != text and redactor(joined) != joined:
            return True
    return False


def _scan_pdf_buffer(buf: bytes, redactor: Callable[[str], str]) -> bool:
    """Scan raw PDF bytes or an inflated stream buffer for redactable secrets."""
    if not buf:
        return False
    if _text_triggers_redaction(buf.decode("latin-1", errors="ignore"), redactor):
        return True
    strings = _extract_pdf_strings(buf)
    if not strings:
        return False
    separated = "\n".join(strings)
    if redactor(separated) != separated:
        return True
    if len(strings) > 1:
        joined = "".join(strings)
        if redactor(joined) != joined:
            return True
    return False


def _extract_stream_dict(window: bytes, stream_pos: int) -> bytes:
    """Return the innermost balanced ``<< ... >>`` dictionary immediately preceding ``stream``."""
    prefix = window[max(0, stream_pos - 4096) : stream_pos].rstrip()
    if not prefix.endswith(b">>"):
        return b""
    depth = 0
    idx = len(prefix) - 2
    while idx >= 0:
        pair = prefix[idx : idx + 2]
        if pair == b">>":
            depth += 1
            idx -= 2
        elif pair == b"<<":
            depth -= 1
            if depth == 0:
                return _decode_pdf_names(prefix[idx:])
            idx -= 2
        else:
            idx -= 1
    return _decode_pdf_names(prefix)


def _extract_stream_filters(stream_dict: bytes) -> list[bytes]:
    if not stream_dict or b"/Filter" not in stream_dict:
        return []
    arr = _PDF_FILTER_ARRAY_RE.search(stream_dict)
    if arr is not None:
        return re.findall(rb"/([A-Za-z0-9._-]+)", arr.group(1))
    single = _PDF_FILTER_SINGLE_RE.search(stream_dict)
    return [single.group(1)] if single is not None else []


def _inflate_flate(body: bytes, max_out: int, *, allow_raw_deflate: bool) -> Optional[bytes]:
    """Bounded FlateDecode inflation. Returns ``None`` when oversized or undecodable."""
    wbits_list = (zlib.MAX_WBITS, -zlib.MAX_WBITS) if allow_raw_deflate else (zlib.MAX_WBITS,)
    for wbits in wbits_list:
        try:
            decomp = zlib.decompressobj(wbits)
            inflated = decomp.decompress(body, max_out + 1)
            if len(inflated) <= max_out and not decomp.unconsumed_tail:
                inflated += decomp.flush(max_out - len(inflated) + 1)
        except zlib.error:
            continue
        if len(inflated) > max_out or decomp.unconsumed_tail:
            return None
        return inflated
    return None


def _decode_runlength(body: bytes, max_out: int) -> Optional[bytes]:
    """Bounded PDF RunLengthDecode per ISO 32000-1 §7.4.5."""
    out = bytearray()
    i, n = 0, len(body)
    while i < n:
        length = body[i]
        i += 1
        if length == 128:
            break
        if length < 128:
            count = length + 1
            if i + count > n or len(out) + count > max_out:
                return None
            out.extend(body[i : i + count])
            i += count
        else:
            count = 257 - length
            if i >= n or len(out) + count > max_out:
                return None
            out.extend(bytes([body[i]]) * count)
            i += 1
    return bytes(out)


def _decode_ascii85(body: bytes, max_out: int) -> Optional[bytes]:
    stripped = body.strip()
    if not stripped.endswith(b"~>"):
        stripped += b"~>"
    try:
        decoded = base64.a85decode(stripped, adobe=True)
    except ValueError:
        return None
    return decoded if len(decoded) <= max_out else None


def _decode_asciihex(body: bytes, max_out: int) -> Optional[bytes]:
    eod = body.find(b">")
    payload = body[:eod] if eod >= 0 else body
    cleaned = b"".join(payload.split())
    if len(cleaned) % 2:
        cleaned += b"0"
    if len(cleaned) // 2 > max_out:
        return None
    try:
        return bytes.fromhex(cleaned.decode("ascii"))
    except ValueError:
        return None


def _decode_flate_filter(body: bytes, max_out: int) -> Optional[bytes]:
    return _inflate_flate(body, max_out, allow_raw_deflate=True)


_PDF_FILTER_DECODERS: dict[bytes, Callable[[bytes, int], Optional[bytes]]] = {
    b"FlateDecode": _decode_flate_filter,
    b"Fl": _decode_flate_filter,
    b"ASCII85Decode": _decode_ascii85,
    b"A85": _decode_ascii85,
    b"ASCIIHexDecode": _decode_asciihex,
    b"AHx": _decode_asciihex,
    b"RunLengthDecode": _decode_runlength,
    b"RL": _decode_runlength,
}


def _decode_pdf_filter_chain(body: bytes, filters: list[bytes], remaining: int) -> tuple[Optional[bytes], int]:
    """Apply a declared PDF filter chain with strict byte bounds.

    Returns ``(decoded_bytes, new_remaining)`` on success, or ``(None, remaining)`` to fail closed.
    """
    if len(filters) > _MAX_PDF_FILTER_CHAIN_DEPTH:
        return None, remaining
    cur = body
    for flt in filters:
        decoder = _PDF_FILTER_DECODERS.get(flt)
        if decoder is None:
            return None, remaining  # unsupported filter (e.g. LZWDecode, Crypt): fail closed
        decoded = decoder(cur, remaining)
        if decoded is None:
            return None, remaining
        remaining -= len(decoded)
        cur = decoded
    return cur, remaining


def pdf_has_redacted_text(data: bytes, redactor: Callable[[str], str]) -> bool:
    """Return True if raw PDF bytes or bounded decoded streams contain redactable secrets.

    Fails closed (returns True) on encrypted PDFs (including ``#XX`` name-escaped ``/Encrypt``),
    predictor-encoded streams, unsupported/malformed declared filters, or oversized/bomb streams so
    raw bytes are never attached natively when secret scanning cannot complete safely.
    """
    if not data:
        return False
    if b"/Encrypt" in data or b"/Encrypt" in _decode_pdf_names(data):
        return True
    if _scan_pdf_buffer(data, redactor):
        return True
    remaining = _MAX_PDF_SCAN_DECOMPRESS_BYTES
    prev_end = 0
    for end_match in _PDF_ENDSTREAM_RE.finditer(data):
        window = data[prev_end : end_match.start()]
        prev_end = end_match.end()
        # Anchor on the LAST ``stream`` before ``endstream`` so an earlier comment/string
        # ending in ``stream\n`` can never swallow a following compressed stream.
        stream_match = None
        for candidate in _PDF_STREAM_START_RE.finditer(window):
            stream_match = candidate
        if stream_match is None:
            continue
        body = window[stream_match.end() :]
        if body.endswith(b"\r\n"):
            body = body[:-2]
        elif body.endswith(b"\n"):
            body = body[:-1]
        if not body:
            continue
        stream_dict = _extract_stream_dict(window, stream_match.start())
        filters = _extract_stream_filters(stream_dict)
        if filters:
            if _PDF_SUBTYPE_IMAGE_RE.search(stream_dict) and all(f in _PDF_IMAGE_ONLY_FILTERS for f in filters):
                continue
            pred_match = _PDF_PREDICTOR_RE.search(stream_dict)
            if pred_match is not None and int(pred_match.group(1)) > 1:
                return True  # fail closed on PNG/TIFF predictor-encoded streams
            decoded, remaining = _decode_pdf_filter_chain(body, filters, remaining)
            if decoded is None:
                return True  # fail closed on unsupported/malformed/oversized filter stream
            if _scan_pdf_buffer(decoded, redactor):
                return True
        else:
            # Undeclared filter (e.g. minimal synthetic PDF or uncompressed stream): try strict zlib.
            try:
                decomp = zlib.decompressobj()
                inflated = decomp.decompress(body, remaining + 1)
                if len(inflated) <= remaining and not decomp.unconsumed_tail:
                    inflated += decomp.flush(remaining - len(inflated) + 1)
            except zlib.error:
                continue
            if len(inflated) > remaining or decomp.unconsumed_tail:
                return True
            remaining -= len(inflated)
            if _scan_pdf_buffer(inflated, redactor):
                return True
    return False


def is_valid_audio_bytes(data: bytes) -> bool:
    """True when *data* looks like a genuine audio container (not HTML/XML/JSON/BOM text)."""
    if not data or len(data) < 4:
        return False
    # UTF-16LE/BE BOMs (0xFF 0xFE / 0xFE 0xFF) would otherwise trip the 0xFF 0xE0 MP3 sync check.
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return False
    head = data[:64].lstrip().lower()
    if any(head.startswith(prefix) for prefix in _TEXT_SPOOF_PREFIXES):
        return False
    # Reject raw PDF disguised as audio
    if data[:5] == b"%PDF-":
        return False
    if len(data) >= 12 and data[:4] == b"FORM" and data[8:12] in (b"AIFF", b"AIFC"):
        return True
    return sniff_container(data[:64]) is not None
