"""Tests for orchestrator_packet_write tool and orchestrator toolset."""
import json
import importlib
import os
import sys
from pathlib import Path
from unittest.mock import patch
import pytest


# ---------------------------------------------------------------------------
# Task 1: toolset resolution
# ---------------------------------------------------------------------------

def test_orchestrator_toolset_exists():
    from toolsets import resolve_toolset
    tools = resolve_toolset("orchestrator")
    assert "orchestrator_packet_write" in tools, (
        f"Expected 'orchestrator_packet_write' in orchestrator toolset, got: {tools}"
    )


# ---------------------------------------------------------------------------
# Task 2: tool registration and check_fn
# ---------------------------------------------------------------------------

def test_tool_is_registered():
    """orchestrator_packet_write must appear in the tool registry."""
    import tools.orchestrator_packet_tool  # noqa: F401 — side-effect import
    from tools.registry import registry
    entry = registry.get_entry("orchestrator_packet_write")
    assert entry is not None, "orchestrator_packet_write not found in registry"
    assert entry.toolset == "orchestrator"
    assert callable(entry.handler)


def test_tool_schema_has_required_fields():
    import tools.orchestrator_packet_tool as mod
    schema = mod.ORCHESTRATOR_PACKET_WRITE_SCHEMA
    props = schema["parameters"]["properties"]
    assert "filename" in props
    assert "content" in props
    assert "overwrite" in props
    required = schema["parameters"]["required"]
    assert "filename" in required
    assert "content" in required
    assert "overwrite" not in required  # optional, defaults to False


def test_check_fn_excludes_worker_profiles():
    """_orchestrator_profile_only must return False for known worker profiles."""
    import tools.orchestrator_packet_tool as mod
    with patch.dict(os.environ, {"HERMES_PROFILE": "orcheapworker"}):
        assert not mod._orchestrator_profile_only(), (
            "Worker profile 'orcheapworker' must be excluded"
        )
    with patch.dict(os.environ, {"HERMES_PROFILE": "evidence-worker"}):
        assert not mod._orchestrator_profile_only(), (
            "Worker profile 'evidence-worker' must be excluded"
        )
    with patch.dict(os.environ, {"HERMES_PROFILE": "hs"}):
        assert mod._orchestrator_profile_only(), (
            "Orchestrator profile 'hs' must be allowed"
        )


def test_check_fn_allows_unset_profile():
    """An unset HERMES_PROFILE (dev/test) must be allowed by check_fn."""
    import tools.orchestrator_packet_tool as mod
    env = {k: v for k, v in os.environ.items() if k != "HERMES_PROFILE"}
    with patch.dict(os.environ, env, clear=True):
        assert mod._orchestrator_profile_only()


# ---------------------------------------------------------------------------
# Task 4: happy-path write
# ---------------------------------------------------------------------------

@pytest.fixture()
def packet_dir(tmp_path, monkeypatch):
    """Redirect all packet writes to a temp directory.

    Sets both HERMES_HOME and HERMES_PACKET_DIR so the canonical-base
    confinement check accepts the temp location.
    """
    hermes = tmp_path / ".hermes"
    hermes.mkdir()
    d = hermes / "cc-packets"
    d.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes))
    monkeypatch.setenv("HERMES_PACKET_DIR", str(d))
    return d


def test_happy_path_write(packet_dir):
    """Writing a valid packet returns filename, path, sha256, size_bytes."""
    import tools.orchestrator_packet_tool as mod
    result = json.loads(mod.orchestrator_packet_write(
        filename="pkt_20260425_081637_1519.md",
        content="# Dispatch Packet\n\npacket_id: pkt_20260425_081637_1519\n",
    ))
    assert result.get("filename") == "pkt_20260425_081637_1519.md"
    assert result.get("path", "").endswith("pkt_20260425_081637_1519.md")
    assert len(result.get("sha256", "")) == 64, "sha256 must be a 64-char hex string"
    assert isinstance(result.get("size_bytes"), int)
    assert result.get("size_bytes") > 0
    assert "error" not in result


def test_file_is_written_to_disk(packet_dir):
    import tools.orchestrator_packet_tool as mod
    content = "hello packet\n"
    mod.orchestrator_packet_write(
        filename="pkt_20260425_081637_1519.md",
        content=content,
    )
    written = (packet_dir / "pkt_20260425_081637_1519.md").read_text()
    assert written == content


def test_sha256_matches_content(packet_dir):
    import hashlib
    import tools.orchestrator_packet_tool as mod
    content = "sha256 check content"
    result = json.loads(mod.orchestrator_packet_write(
        filename="pkt_sha_check.md",
        content=content,
    ))
    expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert result["sha256"] == expected


def test_packet_dir_created_if_missing(tmp_path, monkeypatch):
    """The tool must create the packet directory if it does not exist."""
    hermes = tmp_path / ".hermes"
    hermes.mkdir()
    d = hermes / "new_dir" / "cc-packets"
    assert not d.exists()
    monkeypatch.setenv("HERMES_HOME", str(hermes))
    monkeypatch.setenv("HERMES_PACKET_DIR", str(d))
    import tools.orchestrator_packet_tool as mod
    result = json.loads(mod.orchestrator_packet_write(
        filename="pkt_20260425_081637_1519.md",
        content="content",
    ))
    assert "error" not in result
    assert d.exists()


# ---------------------------------------------------------------------------
# Task 5: security guards
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_filename", [
    "../etc/passwd",
    "/etc/passwd",
    "subdir/file.md",
    "~/secrets.md",
    "file name with spaces.md",
    "file!@#.md",
    "",
    "." * 80 + ".md",               # basename exceeds 72-char limit
    "noextension",                   # no dot → extension group never matches
    "file..double_dot.md",           # second dot in basename fails regex
])
def test_path_traversal_and_invalid_filename_rejected(packet_dir, bad_filename):
    """All path traversal attempts and invalid filenames must return an error."""
    import tools.orchestrator_packet_tool as mod
    result = json.loads(mod.orchestrator_packet_write(
        filename=bad_filename,
        content="content",
    ))
    assert "error" in result, (
        f"Expected error for filename={bad_filename!r}, got: {result}"
    )
    # Must not create any file
    assert not any(packet_dir.iterdir()), (
        f"File was written despite invalid filename {bad_filename!r}"
    )


def test_size_limit_enforced(packet_dir, monkeypatch):
    """Content exceeding HERMES_PACKET_MAX_BYTES must be rejected."""
    import importlib
    monkeypatch.setenv("HERMES_PACKET_MAX_BYTES", "100")
    import tools.orchestrator_packet_tool as mod
    importlib.reload(mod)
    result = json.loads(mod.orchestrator_packet_write(
        filename="pkt_20260425_081637_1519.md",
        content="x" * 101,
    ))
    assert "error" in result
    assert "exceeds" in result["error"].lower()
    assert not any(packet_dir.iterdir())


def test_overwrite_false_blocks_second_write(packet_dir):
    """Second write to the same filename with overwrite=False must fail."""
    import tools.orchestrator_packet_tool as mod
    mod.orchestrator_packet_write(
        filename="pkt_20260425_081637_1519.md",
        content="first write",
    )
    result = json.loads(mod.orchestrator_packet_write(
        filename="pkt_20260425_081637_1519.md",
        content="second write",
    ))
    assert "error" in result
    assert "already exists" in result["error"]
    # Original content must be intact
    assert (packet_dir / "pkt_20260425_081637_1519.md").read_text() == "first write"


def test_overwrite_true_replaces_file(packet_dir):
    """With overwrite=True, an existing packet is replaced."""
    import tools.orchestrator_packet_tool as mod
    mod.orchestrator_packet_write(
        filename="pkt_20260425_081637_1519.md",
        content="first write",
    )
    result = json.loads(mod.orchestrator_packet_write(
        filename="pkt_20260425_081637_1519.md",
        content="second write",
        overwrite=True,
    ))
    assert "error" not in result
    assert (packet_dir / "pkt_20260425_081637_1519.md").read_text() == "second write"


# ---------------------------------------------------------------------------
# Default packet directory: ~/.local/state/life/cc-packets/
# ---------------------------------------------------------------------------

def test_default_packet_dir_is_xdg_state_life(monkeypatch):
    """_packet_dir() must return ~/.local/state/life/cc-packets/ when no env vars are set."""
    import tools.orchestrator_packet_tool as mod
    env = {k: v for k, v in os.environ.items() if k not in ("HERMES_PACKET_DIR", "HERMES_HOME")}
    with patch.dict(os.environ, env, clear=True):
        result = mod._packet_dir()
    expected = Path.home() / ".local" / "state" / "life" / "cc-packets"
    assert result == expected, f"Expected {expected}, got {result}"


def test_raw_packet_dir_default_matches_packet_dir(monkeypatch):
    """_raw_packet_dir() and _packet_dir() must agree on the default."""
    import tools.orchestrator_packet_tool as mod
    env = {k: v for k, v in os.environ.items() if k not in ("HERMES_PACKET_DIR", "HERMES_HOME")}
    with patch.dict(os.environ, env, clear=True):
        raw = mod._raw_packet_dir()
        resolved = mod._packet_dir()
    assert raw == resolved, f"_raw_packet_dir={raw} != _packet_dir={resolved}"


def test_hermes_packet_dir_env_overrides_default(tmp_path, monkeypatch):
    """HERMES_PACKET_DIR inside HERMES_HOME must override the default."""
    import tools.orchestrator_packet_tool as mod
    hermes = tmp_path / ".hermes"
    hermes.mkdir()
    override = hermes / "custom-packets"
    override.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes))
    monkeypatch.setenv("HERMES_PACKET_DIR", str(override))
    assert mod._packet_dir() == override.resolve()


def test_no_base64_writer_in_native_path():
    """The native orchestrator_packet_write path must not invoke hermes_packet_writer.py."""
    import inspect
    import tools.orchestrator_packet_tool as mod
    source = inspect.getsource(mod)
    assert "base64" not in source, (
        "base64 must not appear in orchestrator_packet_tool — native path must not use the compat writer"
    )
    assert "hermes_packet_writer" not in source, (
        "hermes_packet_writer must not be imported or called from the native packet tool"
    )


def test_result_never_contains_content(packet_dir):
    """The JSON result must never echo back the content field."""
    import tools.orchestrator_packet_tool as mod
    secret = "SECRET_PAYLOAD_DATA"
    result_str = mod.orchestrator_packet_write(
        filename="pkt_20260425_081637_1519.md",
        content=secret,
    )
    assert secret not in result_str, (
        "content must never appear in the tool return value"
    )


def test_symlink_target_is_rejected(packet_dir, tmp_path):
    """A symlink planted at the target path must be rejected, not followed."""
    import tools.orchestrator_packet_tool as mod
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("original")
    symlink_path = packet_dir / "pkt_symlink_attack.md"
    symlink_path.symlink_to(outside_file)

    result = json.loads(mod.orchestrator_packet_write(
        filename="pkt_symlink_attack.md",
        content="ATTACKER_PAYLOAD",
    ))
    assert "error" in result, (
        f"Expected error when target is a symlink, got: {result}"
    )
    assert outside_file.read_text() == "original", (
        "Symlink follow must not have written to the target outside pkt_dir"
    )


def test_symlink_target_with_overwrite_is_still_rejected(packet_dir, tmp_path):
    """overwrite=True must not bypass the symlink rejection."""
    import tools.orchestrator_packet_tool as mod
    outside_file = tmp_path / "outside2.txt"
    outside_file.write_text("untouched")
    (packet_dir / "pkt_overwrite_sym.md").symlink_to(outside_file)

    result = json.loads(mod.orchestrator_packet_write(
        filename="pkt_overwrite_sym.md",
        content="PAYLOAD",
        overwrite=True,
    ))
    assert "error" in result
    assert outside_file.read_text() == "untouched"


# ---------------------------------------------------------------------------
# Finding 1: canonical base confinement + symlinked parent rejection
# ---------------------------------------------------------------------------

def test_symlink_parent_dir_is_rejected(tmp_path, monkeypatch):
    """A symlink in the packet directory's ancestor chain must be rejected."""
    hermes = tmp_path / ".hermes"
    hermes.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes))

    # Plant a symlink where the packet dir would live: hermes/link -> real_dir
    real_dir = tmp_path / "real_packets"
    real_dir.mkdir()
    link = hermes / "cc-packets-link"
    link.symlink_to(real_dir)  # the directory itself is a symlink

    monkeypatch.setenv("HERMES_PACKET_DIR", str(link))
    import tools.orchestrator_packet_tool as mod
    result = json.loads(mod.orchestrator_packet_write(
        filename="pkt_attack.md",
        content="ATTACKER_PAYLOAD",
    ))
    assert "error" in result, f"Expected error for symlinked packet dir, got: {result}"
    assert "symlink" in result["error"].lower()
    # Real directory must be untouched
    assert not any(real_dir.iterdir())


def test_in_base_symlinked_parent_is_rejected(tmp_path, monkeypatch):
    """Symlinked HERMES_PACKET_DIR that resolves INSIDE HERMES_HOME must still be rejected.

    This is the confinement gap fixed in round-4: _packet_dir() calls .resolve()
    which eliminates symlinks before _symlinked_component sees the path.  The fix
    checks _raw_packet_dir() (pre-resolution) so a symlink resolving inside the
    allowed base is still caught.
    """
    hermes = tmp_path / ".hermes"
    hermes.mkdir()
    real_packets = hermes / "real-packets"
    real_packets.mkdir()
    link = hermes / "cc-link"
    link.symlink_to(real_packets)  # symlink resolves INSIDE HERMES_HOME

    monkeypatch.setenv("HERMES_HOME", str(hermes))
    monkeypatch.setenv("HERMES_PACKET_DIR", str(link))

    import tools.orchestrator_packet_tool as mod
    result = json.loads(mod.orchestrator_packet_write(
        filename="pkt_inbase_attack.md",
        content="INBASE_PAYLOAD",
    ))
    assert "error" in result, (
        f"Expected error for in-base symlinked packet dir, got: {result}"
    )
    assert "symlink" in result["error"].lower()
    assert not any(real_packets.iterdir()), "No file must be written to the real target"


def test_hostile_env_override_outside_hermes_home_is_rejected(tmp_path, monkeypatch):
    """HERMES_PACKET_DIR pointing outside HERMES_HOME must be rejected."""
    hermes = tmp_path / ".hermes"
    hermes.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes))

    outside = tmp_path / "outside_dir"
    outside.mkdir()
    monkeypatch.setenv("HERMES_PACKET_DIR", str(outside))

    import tools.orchestrator_packet_tool as mod
    result = json.loads(mod.orchestrator_packet_write(
        filename="pkt_escape.md",
        content="ESCAPE_PAYLOAD",
    ))
    assert "error" in result, f"Expected error for hostile HERMES_PACKET_DIR, got: {result}"
    assert not any(outside.iterdir()), "No file must be written to the hostile directory"
