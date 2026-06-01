import json

from tools import youtube_music_digging_tool as tool


def test_parse_artist_title_preserves_identity_parentheses():
    parsed = tool._parse_track_info({
        "title": "Nujabes - Luv(sic) pt3 [Official Video]",
        "uploader": "Uploader",
        "upload_date": "20050101",
    })

    assert parsed.artist == "Nujabes"
    assert parsed.title == "Luv(sic) pt3"
    assert parsed.date == "2005"
    assert parsed.inferred is True


def test_parse_title_cleans_spaces_left_by_removed_noise_terms():
    parsed = tool._parse_track_info({
        "title": "Rick Astley - Never Gonna Give You Up (Official Video) (4K Remaster)",
        "artist": "Rick Astley",
        "uploader": "Rick Astley",
        "upload_date": "20091025",
    })

    assert parsed.artist == "Rick Astley"
    assert parsed.title == "Never Gonna Give You Up (Remaster)"
    assert parsed.date == "2009"
    assert parsed.inferred is True


def test_preflight_blocks_before_download_when_google_auth_bad(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    folder = hermes_home / "integrations"
    folder.mkdir(parents=True)
    (folder / "youtube-audio-google-drive-folder.txt").write_text("folder123", encoding="utf-8")
    monkeypatch.setattr(tool, "get_hermes_home", lambda: str(hermes_home))
    monkeypatch.setattr(tool, "_drive_auth_ok", lambda: (False, "REFRESH_FAILED"))

    called = False

    def fake_download(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("download should not run when Drive auth is bad")

    monkeypatch.setattr(tool, "_download_mp3", fake_download)

    result = json.loads(tool.youtube_music_dig("https://youtu.be/abc123"))

    assert result["success"] is False
    assert result["stage"] == "drive_preflight"
    assert result["error"] == "google_auth_required"
    assert called is False


def test_duplicate_skips_tagging_and_upload(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    folder = hermes_home / "integrations"
    folder.mkdir(parents=True)
    (folder / "youtube-audio-google-drive-folder.txt").write_text("folder123", encoding="utf-8")
    monkeypatch.setattr(tool, "get_hermes_home", lambda: str(hermes_home))
    monkeypatch.setattr(tool, "_drive_auth_ok", lambda: (True, "AUTHENTICATED"))

    mp3 = tmp_path / "download.mp3"
    mp3.write_bytes(b"ID3")
    monkeypatch.setattr(
        tool,
        "_download_mp3",
        lambda url, workdir: (mp3, {"title": "Artist - Title", "uploader": "Uploader"}),
    )
    monkeypatch.setattr(
        tool,
        "_find_existing_drive_file",
        lambda folder_id, filename: {"id": "file1", "name": filename, "webViewLink": "https://drive.example/file1"},
    )

    def fail(*args, **kwargs):
        raise AssertionError("tagging/upload should not run for duplicate")

    monkeypatch.setattr(tool, "_embed_id3", fail)
    monkeypatch.setattr(tool, "_upload_drive", fail)

    result = json.loads(tool.youtube_music_dig("https://youtu.be/abc123"))

    assert result["success"] is True
    assert result["status"] == "duplicate"
    assert result["filename"] == "Artist - Title.mp3"


def test_drive_query_escapes_single_quotes(monkeypatch):
    seen = {}

    class FakeExecute:
        def execute(self):
            return {"files": []}

    class FakeFiles:
        def list(self, **kwargs):
            seen.update(kwargs)
            return FakeExecute()

    class FakeService:
        def files(self):
            return FakeFiles()

    monkeypatch.setattr(tool, "_build_drive_service", lambda: FakeService())
    assert tool._find_existing_drive_file("folder", "Artist's - Title.mp3") is None
    assert "Artist\\'s - Title.mp3" in seen["q"]


def test_retryable_download_failure_is_queued(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    folder = hermes_home / "integrations"
    folder.mkdir(parents=True)
    (folder / "youtube-audio-google-drive-folder.txt").write_text("folder123", encoding="utf-8")
    monkeypatch.setattr(tool, "get_hermes_home", lambda: str(hermes_home))
    monkeypatch.setattr(tool, "_drive_auth_ok", lambda: (True, "AUTHENTICATED"))

    def fail_download(url, workdir):
        raise RuntimeError("Sign in to confirm you’re not a bot")

    monkeypatch.setattr(tool, "_download_mp3", fail_download)

    result = json.loads(tool.youtube_music_dig("https://youtu.be/abc123"))

    assert result["success"] is True
    assert result["status"] == "queued"
    assert result["retry"] == "automatic"
    assert result["queue_id"] == 1

    con = tool._queue_connect()
    row = con.execute("SELECT url, status, last_stage FROM pending_music_dig").fetchone()
    assert row["url"] == "https://youtu.be/abc123"
    assert row["status"] == "pending"
    assert row["last_stage"] == "download"


def test_premium_member_error_tries_remote_fallback(monkeypatch, tmp_path):
    local_mp3 = tmp_path / "remote.mp3"
    local_mp3.write_bytes(b"ID3")
    monkeypatch.setattr(tool, "_remote_host", lambda: "chriss-macbook-pro")

    def fail_local(url, workdir):
        raise RuntimeError("This video is only available to Music Premium members")

    monkeypatch.setattr(tool, "_download_mp3_local", fail_local)
    monkeypatch.setattr(tool, "_download_mp3_remote", lambda url, workdir: (local_mp3, {"title": "Artist - Premium Song"}))

    mp3, info = tool._download_mp3("https://youtu.be/abc123", tmp_path)

    assert mp3 == local_mp3
    assert info["title"] == "Artist - Premium Song"


def test_premium_member_error_is_not_auto_queued():
    assert tool._is_retryable_download_error("This video is only available to Music Premium members") is False
    assert tool._is_retryable_download_error("local yt-dlp failed with premium-only error; remote fallback also failed: HTTP Error 403") is False


def test_queue_upsert_deduplicates_active_url(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setattr(tool, "get_hermes_home", lambda: str(hermes_home))

    first = tool._queue_upsert("https://youtu.be/abc123", stage="download", error="download_failed", detail="ssh: timeout")
    second = tool._queue_upsert("https://youtu.be/abc123", stage="download", error="download_failed", detail="ssh: timeout again")

    assert first["id"] == second["id"]
    assert first["duplicate"] is False
    assert second["duplicate"] is True
    con = tool._queue_connect()
    count = con.execute("SELECT count(*) AS c FROM pending_music_dig").fetchone()["c"]
    assert count == 1


def test_remote_auth_args_default_to_chrome_browser(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setattr(tool, "get_hermes_home", lambda: str(hermes_home))

    assert tool._remote_ytdlp_auth_args() == "--cookies-from-browser chrome"


def test_remote_auth_args_uses_configured_browser_profile(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    integrations = hermes_home / "integrations"
    integrations.mkdir(parents=True)
    (integrations / "youtube-dig-remote-browser.txt").write_text("chrome:Profile 2", encoding="utf-8")
    monkeypatch.setattr(tool, "get_hermes_home", lambda: str(hermes_home))

    assert tool._remote_ytdlp_auth_args() == "--cookies-from-browser 'chrome:Profile 2'"


def test_remote_auth_args_prefers_configured_cookie_file(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    integrations = hermes_home / "integrations"
    integrations.mkdir(parents=True)
    (integrations / "youtube-dig-remote-browser.txt").write_text("chrome:Profile 2", encoding="utf-8")
    (integrations / "youtube-dig-remote-cookie-file.txt").write_text("$HOME/.hermes/youtube-worker/youtube-cookies.txt", encoding="utf-8")
    monkeypatch.setattr(tool, "get_hermes_home", lambda: str(hermes_home))

    assert tool._remote_ytdlp_auth_args() == "--cookies '$HOME/.hermes/youtube-worker/youtube-cookies.txt'"


def test_retry_pending_marks_success_done(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setattr(tool, "get_hermes_home", lambda: str(hermes_home))
    queued = tool._queue_upsert("https://youtu.be/abc123", stage="download", error="download_failed", detail="ssh: timeout")

    def fake_dig(url, queue_on_retryable=True):
        assert queue_on_retryable is False
        return json.dumps({
            "success": True,
            "status": "uploaded",
            "artist": "Artist",
            "title": "Title",
            "filename": "Artist - Title.mp3",
            "source": url,
            "drive": {"id": "drive1", "webViewLink": "https://drive.example/drive1"},
        })

    monkeypatch.setattr(tool, "youtube_music_dig", fake_dig)

    result = json.loads(tool.retry_pending_music_digs(limit=5))

    assert result["processed"] == 1
    assert result["results"][0]["queue_id"] == queued["id"]
    con = tool._queue_connect()
    row = con.execute("SELECT status, attempt_count, filename, drive_file_id FROM pending_music_dig WHERE id = ?", (queued["id"],)).fetchone()
    assert row["status"] == "done"
    assert row["attempt_count"] == 1
    assert row["filename"] == "Artist - Title.mp3"
    assert row["drive_file_id"] == "drive1"
