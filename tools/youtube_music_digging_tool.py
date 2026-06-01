"""YouTube music digging workflow tool.

Downloads a YouTube/YouTube Music link, converts it to MP3, embeds basic ID3
metadata, checks a configured Google Drive inbox for duplicates, and uploads the
finished file. The tool is intentionally conservative: if Drive auth is not
healthy, it refuses before touching YouTube or creating temporary media files.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests
from hermes_constants import get_hermes_home
from tools.registry import registry


TOOL_NAME = "youtube_music_dig"
FOLDER_CONFIG = "integrations/youtube-audio-google-drive-folder.txt"
COOKIE_FILE = "secrets/youtube-cookies.txt"
REMOTE_HOST_CONFIG = "integrations/youtube-dig-remote-host.txt"
REMOTE_COOKIE_FILE_CONFIG = "integrations/youtube-dig-remote-cookie-file.txt"
REMOTE_BROWSER_CONFIG = "integrations/youtube-dig-remote-browser.txt"
CACHE_DIR = "media_cache/youtube-audio"
QUEUE_DB = "music-digging/pending.db"
MAX_QUEUE_ATTEMPTS = 10


YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
}


NOISE_PATTERNS = [
    r"\bOfficial\s+(Music\s+)?Video\b",
    r"\bOfficial\s+MV\b",
    r"\bOfficial\s+Audio\b",
    r"\bOfficial\s+Visualizer\b",
    r"\bMusic\s+Video\b",
    r"\bLyric(s)?\s+Video\b",
    r"\bLyrics?\b",
    r"\b가사\b",
    r"\b한글\s*자막\b",
    r"\b자막\b",
    r"\bHD\b",
    r"\b4K\b",
]


SCHEMA = {
    "name": TOOL_NAME,
    "description": (
        "Process a YouTube/YouTube Music music link for Chris's digging channel: "
        "preflight Google Drive auth, download/convert to MP3, embed basic ID3 "
        "metadata, skip duplicates by Drive filename, and upload to the configured "
        "Drive inbox folder."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "YouTube or YouTube Music URL to process.",
            },
        },
        "required": ["url"],
    },
}


RETRY_SCHEMA = {
    "name": "youtube_music_retry_pending",
    "description": "Retry queued YouTube music digging requests that were deferred because YouTube or the remote fallback worker was temporarily unavailable.",
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "Maximum pending items to process, default 5, max 25."}
        },
    },
}


@dataclass
class TrackInfo:
    artist: str
    title: str
    album: str = ""
    date: str = ""
    thumbnail: str = ""
    inferred: bool = False


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False)


def _home_path(relative: str) -> Path:
    return Path(get_hermes_home()) / relative


def _queue_db_path() -> Path:
    path = _home_path(QUEUE_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _queue_connect() -> sqlite3.Connection:
    con = sqlite3.connect(str(_queue_db_path()), timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")
    _ensure_queue_schema(con)
    return con


def _ensure_queue_schema(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS pending_music_dig (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            source_platform TEXT DEFAULT '',
            source_chat_id TEXT DEFAULT '',
            source_thread_id TEXT DEFAULT '',
            source_user_id TEXT DEFAULT '',
            source_user_name TEXT DEFAULT '',
            requested_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            last_attempt_at INTEGER,
            last_error TEXT DEFAULT '',
            last_stage TEXT DEFAULT '',
            filename TEXT DEFAULT '',
            artist TEXT DEFAULT '',
            title TEXT DEFAULT '',
            drive_file_id TEXT DEFAULT '',
            drive_link TEXT DEFAULT '',
            result_json TEXT DEFAULT ''
        )
        """
    )
    con.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_pending_music_dig_active_url
        ON pending_music_dig(url)
        WHERE status IN ('pending', 'processing')
        """
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_pending_music_dig_status_id ON pending_music_dig(status, id)")
    con.commit()


def _session_value(name: str) -> str:
    try:
        from gateway.session_context import get_session_env
        return get_session_env(name, "") or ""
    except Exception:
        return os.environ.get(name, "") or ""


def _current_source_context() -> dict[str, str]:
    return {
        "source_platform": _session_value("HERMES_SESSION_PLATFORM"),
        "source_chat_id": _session_value("HERMES_SESSION_CHAT_ID"),
        "source_thread_id": _session_value("HERMES_SESSION_THREAD_ID"),
        "source_user_id": _session_value("HERMES_SESSION_USER_ID"),
        "source_user_name": _session_value("HERMES_SESSION_USER_NAME"),
    }


def _queue_upsert(url: str, *, stage: str, error: str, detail: str) -> dict[str, Any]:
    now = int(time.time())
    ctx = _current_source_context()
    with _queue_connect() as con:
        existing = con.execute(
            "SELECT * FROM pending_music_dig WHERE url = ? AND status IN ('pending', 'processing') ORDER BY id LIMIT 1",
            (url,),
        ).fetchone()
        if existing:
            con.execute(
                """
                UPDATE pending_music_dig
                SET status = 'pending', updated_at = ?, last_stage = ?, last_error = ?
                WHERE id = ?
                """,
                (now, stage, detail or error, existing["id"]),
            )
            con.commit()
            row_id = int(existing["id"])
            duplicate = True
        else:
            cur = con.execute(
                """
                INSERT INTO pending_music_dig (
                    url, status, source_platform, source_chat_id, source_thread_id,
                    source_user_id, source_user_name, requested_at, updated_at,
                    last_stage, last_error
                ) VALUES (?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    url,
                    ctx["source_platform"],
                    ctx["source_chat_id"],
                    ctx["source_thread_id"],
                    ctx["source_user_id"],
                    ctx["source_user_name"],
                    now,
                    now,
                    stage,
                    detail or error,
                ),
            )
            con.commit()
            row_id = int(cur.lastrowid)
            duplicate = False
    return {"id": row_id, "duplicate": duplicate, **ctx}


def _is_retryable_download_error(detail: str) -> bool:
    lowered = (detail or "").lower()
    if "music premium" in lowered or "premium members" in lowered or "premium-only" in lowered:
        return False
    retry_markers = [
        "not a bot",
        "sign in to confirm",
        "login_required",
        "remote fallback also failed",
        "connection timed out",
        "operation timed out",
        "could not resolve hostname",
        "no route to host",
        "permission denied",
        "connection refused",
        "ssh:",
        "scp",
        "http error 403",
        "http error 429",
        "temporarily unavailable",
    ]
    return any(marker in lowered for marker in retry_markers)


def _queued_result(url: str, *, stage: str, error: str, detail: str) -> str:
    queued = _queue_upsert(url, stage=stage, error=error, detail=detail)
    return _json({
        "success": True,
        "status": "queued",
        "queue_id": queued["id"],
        "queue_duplicate": queued["duplicate"],
        "stage": stage,
        "error": error,
        "detail": detail,
        "source": url,
        "retry": "automatic",
    })


def _claim_pending(limit: int) -> list[sqlite3.Row]:
    now = int(time.time())
    with _queue_connect() as con:
        con.execute("BEGIN IMMEDIATE")
        rows = con.execute(
            """
            SELECT * FROM pending_music_dig
            WHERE status = 'pending' AND attempt_count < ?
            ORDER BY id
            LIMIT ?
            """,
            (MAX_QUEUE_ATTEMPTS, limit),
        ).fetchall()
        ids = [int(r["id"]) for r in rows]
        if ids:
            placeholders = ",".join("?" for _ in ids)
            con.execute(
                f"""
                UPDATE pending_music_dig
                SET status = 'processing', updated_at = ?, last_attempt_at = ?, attempt_count = attempt_count + 1
                WHERE id IN ({placeholders})
                """,
                (now, now, *ids),
            )
        con.commit()
    return rows


def _finish_queue_item(row_id: int, result: dict[str, Any]) -> None:
    now = int(time.time())
    status = result.get("status")
    success = bool(result.get("success"))
    result_json = json.dumps(result, ensure_ascii=False)
    if success and status in {"uploaded", "duplicate"}:
        drive = result.get("drive") or result.get("existing") or {}
        with _queue_connect() as con:
            con.execute(
                """
                UPDATE pending_music_dig
                SET status = 'done', updated_at = ?, last_error = '', last_stage = '',
                    filename = ?, artist = ?, title = ?, drive_file_id = ?, drive_link = ?, result_json = ?
                WHERE id = ?
                """,
                (
                    now,
                    result.get("filename", ""),
                    result.get("artist", ""),
                    result.get("title", ""),
                    drive.get("id", ""),
                    drive.get("webViewLink", ""),
                    result_json,
                    row_id,
                ),
            )
            con.commit()
        return

    detail = str(result.get("detail") or result.get("error") or "")
    retryable = result.get("status") == "queued" or (result.get("stage") == "download" and _is_retryable_download_error(detail))
    with _queue_connect() as con:
        row = con.execute("SELECT attempt_count FROM pending_music_dig WHERE id = ?", (row_id,)).fetchone()
        attempts = int(row["attempt_count"]) if row else MAX_QUEUE_ATTEMPTS
        next_status = "pending" if retryable and attempts < MAX_QUEUE_ATTEMPTS else "failed"
        con.execute(
            """
            UPDATE pending_music_dig
            SET status = ?, updated_at = ?, last_stage = ?, last_error = ?, result_json = ?
            WHERE id = ?
            """,
            (next_status, now, result.get("stage", ""), detail, result_json, row_id),
        )
        con.commit()


def retry_pending_music_digs(limit: int = 5) -> str:
    limit = max(1, min(int(limit or 5), 25))
    rows = _claim_pending(limit)
    results: list[dict[str, Any]] = []
    for row in rows:
        result = json.loads(youtube_music_dig(row["url"], queue_on_retryable=False))
        _finish_queue_item(int(row["id"]), result)
        results.append({"queue_id": int(row["id"]), **result})
    return _json({"success": True, "processed": len(results), "results": results})


def _skill_python() -> str:
    hermes_home = Path(get_hermes_home())
    venv_python = hermes_home / "hermes-agent" / "venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return os.environ.get("HERMES_PYTHON") or shutil.which("python3") or "python3"


def _google_skill_dir() -> Path:
    """Return the Google Workspace skill dir with the full Drive CLI surface.

    Chris's runtime can have an older user-installed google-workspace skill under
    ~/.hermes/skills that only supports Drive search. Prefer the bundled Hermes
    copy when present because this workflow needs search, upload, and delete/get
    support while still using the same profile-scoped token files.
    """
    bundled = Path(get_hermes_home()) / "hermes-agent" / "skills" / "productivity" / "google-workspace"
    if bundled.exists():
        return bundled
    return Path(get_hermes_home()) / "skills" / "productivity" / "google-workspace"


def _build_drive_service():
    """Build a Drive service via the Google Workspace skill's Python client path.

    Do not route these operations through `gws`: the current token format is
    accepted by the Python client after normalization, while gws rejects tokens
    missing its preferred `type` field. The workflow needs reliable Drive
    search/upload, not CLI-specific token strictness.
    """
    import importlib.util
    import sys

    script = _google_skill_dir() / "scripts" / "google_api.py"
    scripts_dir = str(script.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("_hermes_google_api_for_youtube_music", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load google_api.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_service("drive", "v3")


def _run_google_setup(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess[str]:
    script = _google_skill_dir() / "scripts" / "setup.py"
    return subprocess.run(
        [_skill_python(), str(script), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def _drive_auth_ok() -> tuple[bool, str]:
    proc = _run_google_setup(["--check"])
    output = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0 and "AUTHENTICATED" in output, output


def _drive_folder_id() -> str:
    path = _home_path(FOLDER_CONFIG)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _remote_host() -> str:
    path = _home_path(REMOTE_HOST_CONFIG)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _read_optional_config(relative_path: str) -> str:
    path = _home_path(relative_path)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _remote_ytdlp_auth_args() -> str:
    """Return shell-quoted remote yt-dlp authentication args.

    The remote fallback runs on a trusted residential/browser host. Prefer an
    explicitly exported Netscape cookies file on that host when configured;
    otherwise use a configured browser profile, defaulting to Chrome's default
    profile for Chris's current Mac worker.
    """
    remote_cookie_file = _read_optional_config(REMOTE_COOKIE_FILE_CONFIG)
    if remote_cookie_file:
        return f"--cookies {shlex_quote(remote_cookie_file)}"

    remote_browser = _read_optional_config(REMOTE_BROWSER_CONFIG) or "chrome"
    return f"--cookies-from-browser {shlex_quote(remote_browser)}"


def _run(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, timeout=timeout, check=False)


def _yt_error_text(exc: Exception) -> str:
    return str(exc)


def _looks_like_youtube_bot_challenge(text: str) -> bool:
    lowered = (text or "").lower()
    return "not a bot" in lowered or "sign in to confirm" in lowered or "login_required" in lowered


def _looks_like_youtube_auth_required(text: str) -> bool:
    lowered = (text or "").lower()
    return _looks_like_youtube_bot_challenge(text) or "music premium" in lowered or "premium members" in lowered


def _is_supported_youtube_url(url: str) -> bool:
    try:
        parsed = urlparse(url.strip())
    except Exception:
        return False
    host = (parsed.netloc or "").lower()
    if host not in YOUTUBE_HOSTS:
        return False
    if host.endswith("youtu.be"):
        return bool(parsed.path.strip("/"))
    if parsed.path in {"/watch", "/shorts", "/embed"} or parsed.path.startswith("/watch"):
        return bool(parse_qs(parsed.query).get("v") or parsed.path.startswith(("/shorts/", "/embed/")))
    return host == "music.youtube.com" and bool(parse_qs(parsed.query).get("v"))


def _sanitize_component(value: str) -> str:
    value = re.sub(r"[\\/:*?\"<>|]+", " ", value or "")
    value = re.sub(r"\s+", " ", value).strip(" .-_\t\n")
    return value[:120] or "Unknown"


def _strip_wrappers(text: str) -> tuple[str, bool]:
    inferred = False
    s = text or ""

    def repl(match: re.Match[str]) -> str:
        nonlocal inferred
        inner = match.group(1).strip()
        keep = re.search(r"\b(feat\.?|ft\.?|remix|live|edit|version|ver\.?|remaster|cover)\b", inner, re.I)
        if keep:
            return match.group(0)
        if any(re.search(p, inner, re.I) for p in NOISE_PATTERNS):
            inferred = True
            return " "
        return match.group(0)

    s = re.sub(r"[\[【(（]([^\]】)）]+)[\]】)）]", repl, s)
    for pat in NOISE_PATTERNS:
        ns = re.sub(pat, " ", s, flags=re.I)
        if ns != s:
            inferred = True
            s = ns
    s = re.sub(r"\s*[|｜]\s*(YouTube|MV|Official).*?$", "", s, flags=re.I)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"([\[(（【])\s+", r"\1", s)
    s = re.sub(r"\s+([\])）】])", r"\1", s)
    s = re.sub(r"[\[(（【]\s*[\])）】]", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" -–—|｜")
    return s or text, inferred


def _parse_track_info(info: dict[str, Any]) -> TrackInfo:
    raw_title = str(info.get("track") or info.get("title") or "").strip()
    raw_artist = str(info.get("artist") or "").strip()
    uploader = str(info.get("uploader") or info.get("channel") or "").strip()
    album = str(info.get("album") or "").strip()
    date = str(info.get("release_date") or info.get("upload_date") or "").strip()
    thumbnail = str(info.get("thumbnail") or "").strip()

    cleaned_title, cleaned_changed = _strip_wrappers(raw_title)
    inferred = cleaned_changed

    artist = raw_artist
    title = cleaned_title

    m = re.match(r"^(.{1,120}?)\s+[-–—]\s+(.{1,180})$", cleaned_title)
    if m:
        left, right = m.group(1).strip(), m.group(2).strip()
        if left and right:
            artist, title = left, right
            inferred = True
    elif not artist:
        artist = uploader
        inferred = True

    artist, artist_changed = _strip_wrappers(artist)
    title, title_changed = _strip_wrappers(title)
    inferred = inferred or artist_changed or title_changed or not raw_artist

    return TrackInfo(
        artist=_sanitize_component(artist),
        title=_sanitize_component(title),
        album=_sanitize_component(album) if album else "",
        date=date[:4] if re.fullmatch(r"\d{8}", date) else date,
        thumbnail=thumbnail,
        inferred=inferred,
    )


def _drive_query_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _find_existing_drive_file(folder_id: str, filename: str) -> dict[str, Any] | None:
    query = (
        f"name = '{_drive_query_escape(filename)}' and "
        f"'{_drive_query_escape(folder_id)}' in parents and trashed = false"
    )
    service = _build_drive_service()
    result = service.files().list(
        q=query,
        pageSize=10,
        fields="files(id, name, mimeType, modifiedTime, webViewLink)",
    ).execute()
    files = result.get("files", [])
    if files:
        return files[0]
    return None


def _download_mp3(url: str, workdir: Path) -> tuple[Path, dict[str, Any]]:
    try:
        return _download_mp3_local(url, workdir)
    except Exception as exc:
        detail = _yt_error_text(exc)
        if _looks_like_youtube_auth_required(detail) and _remote_host():
            try:
                return _download_mp3_remote(url, workdir)
            except Exception as remote_exc:
                prefix = "premium-only" if "premium" in detail.lower() else "auth-required"
                raise RuntimeError(f"local yt-dlp failed with {prefix} error; remote fallback also failed: {remote_exc}") from remote_exc
        raise


def _download_mp3_local(url: str, workdir: Path) -> tuple[Path, dict[str, Any]]:
    from yt_dlp import YoutubeDL

    outtmpl = str(workdir / "download.%(ext)s")
    opts: dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
    }
    cookies = _home_path(COOKIE_FILE)
    if cookies.exists():
        opts["cookiefile"] = str(cookies)
    if shutil.which("node"):
        opts["js_runtimes"] = {"node": {}}

    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        prepared = Path(ydl.prepare_filename(info)).with_suffix(".mp3")
    if not prepared.exists():
        candidates = list(workdir.glob("*.mp3"))
        if not candidates:
            raise RuntimeError("MP3 output was not created")
        prepared = candidates[0]
    return prepared, info


def _json_from_output(output: str) -> dict[str, Any]:
    start = output.find("{")
    if start < 0:
        raise RuntimeError("yt-dlp did not return JSON metadata")
    return json.loads(output[start:], strict=False)


def _download_mp3_remote(url: str, workdir: Path) -> tuple[Path, dict[str, Any]]:
    host = _remote_host()
    if not host:
        raise RuntimeError("missing remote fallback host config")
    run_id = f"dig-{uuid.uuid4().hex}"
    remote_dir = f"$HOME/.hermes/youtube-worker/runs/{run_id}"
    remote_python = "$HOME/.hermes/youtube-worker/venv/bin/python"
    remote_ytdlp = "$HOME/.hermes/youtube-worker/venv/bin/yt-dlp"
    ssh_base = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host]
    extractor_args = "youtube:player_client=android"
    auth_args = _remote_ytdlp_auth_args()

    info_cmd = (
        f"{remote_ytdlp} --quiet --no-warnings --skip-download --dump-single-json "
        f"--no-playlist {auth_args} --extractor-args {shlex_quote(extractor_args)} {shlex_quote(url)}"
    )
    info_proc = _run([*ssh_base, info_cmd], timeout=120)
    if info_proc.returncode != 0:
        raise RuntimeError((info_proc.stderr or info_proc.stdout or "remote metadata failed").strip())
    info = _json_from_output(info_proc.stdout)

    download_cmd = (
        f"set -e; mkdir -p {remote_dir}; "
        f"{remote_ytdlp} --quiet --no-warnings --no-playlist "
        f"{auth_args} --extractor-args {shlex_quote(extractor_args)} -f {shlex_quote('bestaudio/best')} "
        f"-o {shlex_quote(remote_dir + '/download.%(ext)s')} {shlex_quote(url)}; "
        f"{remote_python} - <<'PY'\n"
        "from pathlib import Path\n"
        f"p=next((Path.home() / '.hermes/youtube-worker/runs/{run_id}').glob('download.*'))\n"
        "print('REMOTE_FILE:' + str(p))\n"
        "PY"
    )
    dl_proc = _run([*ssh_base, download_cmd], timeout=240)
    if dl_proc.returncode != 0:
        cleanup_cmd = f"rm -rf {remote_dir}"
        _run([*ssh_base, cleanup_cmd], timeout=30)
        raise RuntimeError((dl_proc.stderr or dl_proc.stdout or "remote download failed").strip())

    remote_file = ""
    for line in dl_proc.stdout.splitlines():
        if line.startswith("REMOTE_FILE:"):
            remote_file = line.split(":", 1)[1].strip()
    if not remote_file:
        _run([*ssh_base, f"rm -rf {remote_dir}"], timeout=30)
        raise RuntimeError("remote download did not report output path")

    raw_path = workdir / ("remote" + Path(remote_file).suffix)
    scp_proc = _run(["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", f"{host}:{remote_file}", str(raw_path)], timeout=180)
    _run([*ssh_base, f"rm -rf {remote_dir}"], timeout=30)
    if scp_proc.returncode != 0:
        raise RuntimeError((scp_proc.stderr or scp_proc.stdout or "remote scp failed").strip())

    mp3_path = workdir / "download.mp3"
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for remote fallback conversion")
    ffmpeg_proc = _run([ffmpeg, "-y", "-i", str(raw_path), "-vn", "-codec:a", "libmp3lame", "-b:a", "192k", str(mp3_path)], timeout=180)
    if ffmpeg_proc.returncode != 0 or not mp3_path.exists():
        raise RuntimeError((ffmpeg_proc.stderr or ffmpeg_proc.stdout or "ffmpeg conversion failed").strip())
    return mp3_path, info


def shlex_quote(value: str) -> str:
    import shlex
    return shlex.quote(value)


def _embed_id3(path: Path, info: TrackInfo, source_url: str) -> None:
    from mutagen.easyid3 import EasyID3
    from mutagen.id3 import APIC, COMM, ID3, ID3NoHeaderError
    from mutagen.mp3 import MP3

    try:
        audio = EasyID3(str(path))
    except Exception:
        audio = MP3(str(path), ID3=EasyID3)
        audio.add_tags()
        audio = EasyID3(str(path))

    audio["title"] = info.title
    audio["artist"] = info.artist
    if info.album:
        audio["album"] = info.album
    if info.date:
        audio["date"] = info.date
    audio.save()

    try:
        tags = ID3(str(path))
    except ID3NoHeaderError:
        tags = ID3()
    tags.delall("COMM")
    tags.add(COMM(encoding=3, lang="eng", desc="Source", text=source_url))

    if info.thumbnail:
        try:
            r = requests.get(info.thumbnail, timeout=20)
            r.raise_for_status()
            ctype = r.headers.get("content-type") or "image/jpeg"
            if ctype.startswith("image/"):
                tags.delall("APIC")
                tags.add(APIC(encoding=3, mime=ctype.split(";")[0], type=3, desc="Cover", data=r.content))
        except Exception:
            pass
    tags.save(str(path), v2_version=3)


def _upload_drive(local_path: Path, folder_id: str, filename: str) -> dict[str, Any]:
    import mimetypes
    from googleapiclient.http import MediaFileUpload

    service = _build_drive_service()
    mime = mimetypes.guess_type(str(local_path))[0] or "audio/mpeg"
    media = MediaFileUpload(str(local_path), mimetype=mime, resumable=True)
    result = service.files().create(
        body={"name": filename, "parents": [folder_id]},
        media_body=media,
        fields="id, name, mimeType, webViewLink",
    ).execute()
    return {
        "status": "uploaded",
        "id": result["id"],
        "name": result.get("name", ""),
        "mimeType": result.get("mimeType", ""),
        "webViewLink": result.get("webViewLink", ""),
    }


def youtube_music_dig(url: str, queue_on_retryable: bool = True) -> str:
    url = (url or "").strip()
    if not _is_supported_youtube_url(url):
        return _json({"success": False, "stage": "validation", "error": "unsupported_url", "source": url})

    folder_id = _drive_folder_id()
    if not folder_id:
        return _json({"success": False, "stage": "drive_preflight", "error": "missing_drive_folder_config"})

    ok, detail = _drive_auth_ok()
    if not ok:
        return _json({
            "success": False,
            "stage": "drive_preflight",
            "error": "google_auth_required",
            "detail": detail,
            "source": url,
        })

    cache_root = _home_path(CACHE_DIR)
    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dig-", dir=str(cache_root)) as tmp:
        workdir = Path(tmp)
        try:
            mp3_path, raw_info = _download_mp3(url, workdir)
        except Exception as exc:
            detail = str(exc)
            if queue_on_retryable and _is_retryable_download_error(detail):
                return _queued_result(url, stage="download", error="download_failed", detail=detail)
            return _json({"success": False, "stage": "download", "error": "download_failed", "detail": detail, "source": url})

        info = _parse_track_info(raw_info)
        filename = f"{info.artist} - {info.title}.mp3"

        try:
            existing = _find_existing_drive_file(folder_id, filename)
        except Exception as exc:
            return _json({
                "success": False,
                "stage": "drive_duplicate_check",
                "error": "drive_search_failed",
                "detail": str(exc),
                "source": url,
            })
        if existing:
            return _json({
                "success": True,
                "status": "duplicate",
                "artist": info.artist,
                "title": info.title,
                "album": info.album,
                "year": info.date,
                "filename": filename,
                "source": url,
                "existing": existing,
                "inferred": info.inferred,
            })

        final_path = workdir / filename
        mp3_path.rename(final_path)
        try:
            _embed_id3(final_path, info, url)
        except Exception as exc:
            return _json({"success": False, "stage": "tagging", "error": "metadata_failed", "detail": str(exc), "source": url})

        try:
            uploaded = _upload_drive(final_path, folder_id, filename)
        except Exception as exc:
            return _json({"success": False, "stage": "drive_upload", "error": "drive_upload_failed", "detail": str(exc), "source": url})

        return _json({
            "success": True,
            "status": "uploaded",
            "artist": info.artist,
            "title": info.title,
            "album": info.album,
            "year": info.date,
            "filename": filename,
            "source": url,
            "drive": uploaded,
            "inferred": info.inferred,
        })


def _handle(args: dict[str, Any], **_: Any) -> str:
    return youtube_music_dig(args.get("url", ""))


def _handle_retry(args: dict[str, Any], **_: Any) -> str:
    return retry_pending_music_digs(args.get("limit") or 5)


def check_requirements() -> bool:
    return bool(shutil.which("ffmpeg"))


registry.register(
    name=TOOL_NAME,
    toolset="media",
    schema=SCHEMA,
    handler=_handle,
    check_fn=check_requirements,
    requires_env=[],
    is_async=False,
    emoji="🎧",
)

registry.register(
    name="youtube_music_retry_pending",
    toolset="media",
    schema=RETRY_SCHEMA,
    handler=_handle_retry,
    check_fn=check_requirements,
    requires_env=[],
    is_async=False,
    emoji="🔁",
)
