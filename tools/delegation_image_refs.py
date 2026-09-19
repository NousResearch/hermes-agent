"""Bounded, process-local image metadata for a live child's transcript writer.

Only explicit tool image fields and standalone result markers are inspected. No
network resolution, transcript/prose scraping, sandbox-to-host guesses or pixels
in the mounted delegation logs. File access remains the authenticated fs API's job.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import threading
from pathlib import Path

logger = logging.getLogger(__name__)
MAX_IMAGES = 32
MAX_REF_CHARS = 2048
MAX_REFS_WIRE_BYTES = 24 * 1024
MAX_EVENT_CHARS = 8 * 1024 * 1024
MAX_TEXT_CHARS = 64 * 1024
MAX_NODES = 512
MAX_DEPTH = 8
MAX_NATIVE_BYTES = 4 * 1024 * 1024
MAX_NATIVE_TOTAL_BYTES = 16 * 1024 * 1024
_IMAGE_FIELDS = ('image', 'image_url', 'image_path', 'images', 'image_paths', 'screenshot_path', 'host_image', 'path', 'url')
_CONTAINERS = ('result', 'output', 'content', 'stdout', 'text', 'results')
_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}
_MIME_EXT = {'image/png': '.png', 'image/jpeg': '.jpg', 'image/gif': '.gif',
             'image/webp': '.webp', 'image/bmp': '.bmp'}
_MARKER = re.compile(r'^\s*(?:MEDIA:|Screenshot:|Screenshot path:|Screenshot saved to:?)\s*(.+?)\s*$', re.IGNORECASE)
_DATA_URL = re.compile(r'data:image/[^\s\"\'<>]+', re.IGNORECASE)


def log_preview(value):
    """Do not serialize native blocks into the read-only sandbox-mounted log."""
    if isinstance(value, str):
        # JSON envelopes may contain bare MCP base64 (not a data URL).
        if value.lstrip().startswith(('{', '[')):
            if len(value) > MAX_EVENT_CHARS:
                return '[oversize structured tool payload omitted]'
            try:
                return log_preview(json.loads(value))
            except (ValueError, RecursionError):
                return '[unparseable structured tool payload omitted]'
        return _DATA_URL.sub('[image payload omitted]', value[:MAX_TEXT_CHARS])
    if isinstance(value, (dict, list)):
        # Bounded sanitization; never stringifies an unknown nested image payload.
        budget = [MAX_NODES]

        def clean(obj, depth=0):
            budget[0] -= 1
            if budget[0] < 0 or depth > MAX_DEPTH:
                return '[payload omitted]'
            if isinstance(obj, dict):
                if obj.get('_multimodal') or obj.get('type') in ('image', 'image_url', 'input_image'):
                    return '[image payload omitted]'
                return {str(k)[:128]: clean(v, depth + 1) for k, v in list_items(obj)}
            if isinstance(obj, list):
                return [clean(v, depth + 1) for v in obj[:MAX_NODES]]
            if isinstance(obj, str):
                return _DATA_URL.sub('[image payload omitted]', obj[:MAX_TEXT_CHARS])
            return obj if obj is None or isinstance(obj, (bool, int, float)) else '[payload omitted]'
        return clean(value)
    return value


def list_items(obj):
    # islice avoids allocating a copy of a potentially huge dictionary.
    from itertools import islice
    return islice(obj.items(), MAX_NODES)


class LiveImageRefs:
    def __init__(self, child):
        self.child = child
        self._lock = threading.Lock()
        self._refs: dict[str, str | None] = {}
        self._native_cache: dict[str, str] = {}
        self._native_bytes = 0
        self._revision = 0
        self._saw_image = False
        self._terminal_inflight = 0
        self._terminal_transient = False
        self.truncated = False

    def snapshot(self):
        with self._lock:
            return {'images': list(self._refs), 'images_truncated': self.truncated and self._saw_image, 'image_revision': self._revision}

    def _path(self, value, cwd, local):
        if not isinstance(value, str) or not value:
            return None
        self._saw_image = True
        if len(value) > MAX_REF_CHARS:
            self.truncated = True
            return None
        # URLs (including file:// authorities), UNC and controls are never gallery refs.
        if any(ord(c) < 32 or ord(c) == 127 for c in value) or value.startswith(('//', '\\\\')):
            return None
        scheme = re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', value)
        native_drive = os.name == 'nt' and re.match(r'^[a-zA-Z]:[\\/]', value)
        if scheme and not native_drive:
            self.truncated = True
            return None
        if not local:
            self.truncated = True
            return None  # Native output can still be cached; never reinterpret sandbox paths.
        path = Path(value).expanduser()
        if path.suffix.lower() not in _EXTENSIONS:
            if path.suffix.lower() in ('.svg', '.avif', '.tif', '.tiff', '.ico'):
                self.truncated = True
            return None
        if not path.is_absolute():
            if not cwd or not Path(cwd).is_absolute():
                self.truncated = True
                return None
            path = Path(cwd) / path
        # Lexical normalization, not symlink/file I/O; fs/read-data-url owns read authorization.
        ref = os.path.normpath(str(path))
        if len(ref) > MAX_REF_CHARS:
            self.truncated = True
            return None
        return ref

    def _cache_native(self, value, mime=None):
        if not isinstance(value, str):
            return None
        if value.startswith('data:'):
            header, sep, payload = value.partition(',')
            if not sep or not header.endswith(';base64'):
                return None
            mime, value = header[5:-7], payload
        ext = _MIME_EXT.get(mime) if isinstance(mime, str) else None
        if not ext:
            self._saw_image = True
            self.truncated = True
            return None
        self._saw_image = True
        if len(value) > ((MAX_NATIVE_BYTES + 2) // 3) * 4:
            self.truncated = True
            return None
        if not value.isascii():
            return None
        digest = hashlib.sha256(value.encode('ascii')).hexdigest()
        if digest in self._native_cache:
            return self._native_cache[digest]
        if len(self._native_cache) >= MAX_IMAGES:
            self.truncated = True
            return None
        try:
            data = base64.b64decode(value, validate=True)
            if len(data) > MAX_NATIVE_BYTES or self._native_bytes + len(data) > MAX_NATIVE_TOTAL_BYTES:
                self.truncated = True
                return None
            from gateway.platforms.base import cache_image_from_bytes
            ref = cache_image_from_bytes(data, ext=ext)
        except (ValueError, OSError) as exc:
            logger.debug('Live image cache rejected payload: %s', type(exc).__name__)
            return None
        self._native_bytes += len(data)
        self._native_cache[digest] = ref
        return ref

    def _extract(self, value, cwd, local, *, started):
        refs, native = [], []
        budget = MAX_NODES

        def visit(obj, depth=0, image=False):
            nonlocal budget
            budget -= 1
            if budget < 0 or depth > MAX_DEPTH:
                self.truncated = True
                return
            if isinstance(obj, str):
                if image:
                    ref = self._cache_native(obj) if obj.startswith('data:image/') else self._path(obj, cwd, local)
                    if ref:
                        refs.append(ref)
                elif not started:
                    if len(obj) > MAX_EVENT_CHARS:
                        self.truncated = True
                        return
                    if obj.lstrip().startswith(('{', '[')):
                        try:
                            visit(json.loads(obj), depth + 1)
                        except (ValueError, RecursionError):
                            return
                    else:
                        if len(obj) > MAX_TEXT_CHARS:
                            self.truncated = True
                        for line in obj[:MAX_TEXT_CHARS].splitlines():
                            match = _MARKER.fullmatch(line)
                            if match:
                                ref = match[1]
                                if len(ref) >= 2 and ref[0] in ('\"', "'", '`') and ref[-1] == ref[0]:
                                    ref = ref[1:-1]
                                visit(ref, depth + 1, True)
                return
            if isinstance(obj, list):
                for item in obj[:MAX_NODES]:
                    if budget <= 0:
                        self.truncated = True
                        break
                    visit(item, depth + 1, image)
                if len(obj) > MAX_NODES:
                    self.truncated = True
                return
            if not isinstance(obj, dict):
                return
            kind = obj.get('type')
            if not started and kind in ('image', 'image_url', 'input_image'):
                data = obj.get('image_url')
                if isinstance(data, dict):
                    data = data.get('url')
                source = obj.get('source')
                if isinstance(source, dict) and source.get('type') == 'base64':
                    data, mime = source.get('data'), source.get('media_type')
                else:
                    data = data or obj.get('data')
                    mime = obj.get('mimeType') or obj.get('mime_type')
                native.append((data, mime))
                return
            for key in _IMAGE_FIELDS:
                if key in obj:
                    visit(obj[key], depth + 1, True)
            meta = obj.get('meta')
            if isinstance(meta, dict) and 'screenshot_path' in meta:
                visit(meta['screenshot_path'], depth + 1, True)
            if image:
                # Explicit images: [{path: ...}] containers, not arbitrary page assets.
                for key in ('path', 'url'):
                    if key in obj:
                        visit(obj[key], depth + 1, True)
            elif not started:
                for key in _CONTAINERS:
                    if key in obj:
                        visit(obj[key], depth + 1)

        visit(value)
        if native:
            # Pixel blocks, including crops, are authoritative; ignore embedded questions/meta.
            refs = [ref for data, mime in native if (ref := self._cache_native(data, mime))]
        return refs, bool(native)

    def capture(self, name, value, *, started):
        from tools.terminal_tool import get_session_cwd, resolve_task_overrides
        from tools.image_source import _is_local_terminal_backend

        # Query at event time, not construction time or gateway poll time. Never use
        # the parent's/default task or the gateway's arbitrary process working dir.
        task_id = getattr(self.child, '_current_task_id', None) or getattr(self.child, '_subagent_id', None)
        cwd = get_session_cwd(task_id) if isinstance(task_id, str) and task_id else None
        overrides = resolve_task_overrides(task_id) if isinstance(task_id, str) and task_id else {}
        backend = overrides.get('env_type')
        local = backend == 'local' if backend else _is_local_terminal_backend()
        with self._lock:
            if name == 'terminal':
                if started:
                    self._terminal_inflight += 1
                    self._terminal_transient |= isinstance(value, dict) and bool(value.get('workdir'))
                # A transient workdir (possibly followed by an internal cd) is not
                # the recorded session cwd. Without call ids don't guess a pairing
                # across parallel commands; absolute output refs remain usable.
                if self._terminal_transient:
                    cwd = None
                if not started:
                    self._terminal_inflight = max(0, self._terminal_inflight - 1)
                    if not self._terminal_inflight:
                        self._terminal_transient = False
            refs, native = self._extract(value, cwd, local, started=started)
            if refs:
                self._revision += 1
            # The callback has no call id: do not guess FIFO pairing for parallel tools.
            # Native results supersede provisional input refs from that tool, never
            # other tools' inputs or already completed image outputs.
            source = str(name or '')[:128]
            if native:
                self._refs = {ref: origin for ref, origin in self._refs.items() if origin != source}
            for ref in refs:
                if len(ref) > MAX_REF_CHARS:
                    self.truncated = True
                    continue
                if ref in self._refs:
                    if not started:
                        self._refs[ref] = None
                    continue
                cost = len(json.dumps(ref).encode('utf-8')) + 2
                total = sum(len(json.dumps(r).encode('utf-8')) + 2 for r in self._refs)
                if cost > MAX_REFS_WIRE_BYTES:
                    self.truncated = True
                    continue
                while self._refs and (len(self._refs) >= MAX_IMAGES or total + cost > MAX_REFS_WIRE_BYTES):
                    oldest = next(iter(self._refs))
                    total -= len(json.dumps(oldest).encode('utf-8')) + 2
                    del self._refs[oldest]
                    self.truncated = True
                self._refs[ref] = source if started else None
