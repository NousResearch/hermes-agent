"""R3-S1 extraction regression: pet payload moved to tui_gateway.pet_payload.

Covers the consensus seam contract (epic #78647, target #78630):
- identity-preserving re-export of all 23 members (server namespace binds the
  exact same objects the new module owns),
- handler + watcher liveness through the re-export binding,
- no import cycle in any order,
- byte-verbatim golden-sha regression for the moved span,
- aggressive unit coverage of the moved cluster (payload shape, cache cap,
  clone-on-read, cancel lifecycle, reference-image validation, downscale,
  gen sweep staleness, config-scale fail-open).
"""

from __future__ import annotations

import base64
import hashlib
import pathlib
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from tui_gateway import pet_payload, server  # noqa: E402

GOLDEN_SHA = "99120f354c2675612f796c4d4fb19477f7a9983b1135d5dd0c17471ea3aa59aa"

MEMBERS = [
    "_pet_frame_counts", "_pet_payload_cache_lock", "_pet_payload_cache",
    "_pet_sheet_revision", "_pet_payload_cache_key", "_clone_pet_payload",
    "_pet_row_frame_counts", "_pet_config_scale", "_pet_sprite_payload",
    "_pet_active_selection", "_pet_state_rows", "_pet_gen_root", "_pet_gen_sweep",
    "_pet_png_data_uri", "_pet_cancel_lock", "_pet_cancelled",
    "_PET_REFERENCE_MIME_EXT", "_PET_REFERENCE_MAX_BYTES",
    "_pet_reference_images_from_data_url", "_pet_cancel_arm", "_pet_cancel_request",
    "_pet_is_cancelled", "_pet_cancel_release",
]

WIRE_KEYS = {
    "slug", "displayName", "mime", "spritesheetBase64", "spritesheetRevision",
    "frameW", "frameH", "framesPerState", "framesByState", "framesByRow",
    "loopMs", "scale", "stateRows",
}


@pytest.fixture(autouse=True)
def _reset_pet_state():
    """Pet caches/cancel set now live in pet_payload (post-extraction)."""
    pet_payload._pet_payload_cache.clear()
    pet_payload._pet_cancelled.clear()
    yield


def _png(path, size=(64, 64), color=(200, 80, 80, 255)):
    Image.new("RGBA", size, color).save(path)
    return pathlib.Path(path)


def _fake_pet(path, slug="test-pet", display_name="Test Pet"):
    return SimpleNamespace(
        slug=slug,
        display_name=display_name,
        spritesheet=pathlib.Path(path),
        exists=True,
    )


# ---------------------------------------------------------------------------
# Seam identity: re-export binds the exact same objects (23 members)
# ---------------------------------------------------------------------------

def test_reexport_identity_all_23_members():
    for name in MEMBERS:
        assert name in pet_payload.__dict__, f"{name} missing from pet_payload"
        assert name in vars(server), f"{name} missing from server namespace"
        assert getattr(server, name) is getattr(pet_payload, name), (
            f"{name} re-export is not identity-preserving"
        )


def test_pet_payload_defines_exactly_the_23_members():
    import ast

    src = pathlib.Path(pet_payload.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    defined = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    defined.add(t.id)
    assert set(MEMBERS) <= defined, f"missing definitions: {set(MEMBERS) - defined}"
    # no stray pet-cluster names beyond the adjudicated 23
    stray = {n for n in defined if n.startswith(("_pet", "_PET", "_clone"))} - set(MEMBERS)
    assert not stray, f"unexpected pet-cluster definitions: {stray}"


def test_golden_sha_span_still_verbatim_in_module():
    src = pathlib.Path(pet_payload.__file__).read_text(encoding="utf-8")
    marker = "def _pet_frame_counts"
    assert marker in src
    span = src[src.index(marker):]
    digest = hashlib.sha256(span.encode("utf-8")).hexdigest()
    assert digest == GOLDEN_SHA, f"moved span drifted: {digest}"


def test_lock_and_cancel_state_same_object_across_boundary():
    # The lock objects must be the SAME objects across the module boundary so
    # concurrency semantics survive (consensus section 4.3).
    assert server._pet_payload_cache_lock is pet_payload._pet_payload_cache_lock
    assert server._pet_cancel_lock is pet_payload._pet_cancel_lock
    assert server._pet_payload_cache is pet_payload._pet_payload_cache
    assert server._pet_cancelled is pet_payload._pet_cancelled


# ---------------------------------------------------------------------------
# Import cycle: all orders clean
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("order", [
    "import tui_gateway.pet_payload; import tui_gateway.server; import tui_gateway.methods_session",
    "import tui_gateway.methods_session; import tui_gateway.server; import tui_gateway.pet_payload",
    "import tui_gateway.server; import tui_gateway.pet_payload; import tui_gateway.methods_session",
])
def test_no_import_cycle_any_order(order):
    code = f"{order}; print('ok')"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=str(pathlib.Path(__file__).parents[2]),
        timeout=120,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined
    assert "ok" in combined


# ---------------------------------------------------------------------------
# Handler + watcher liveness through the re-export binding
# ---------------------------------------------------------------------------

def test_pet_info_handler_live_through_reexport():
    # Rebound handler's bare _pet_active_selection/_pet_sprite_payload resolve
    # through server.py's namespace (HandlerRegistry.install rebinds __globals__).
    resp = server._methods["pet.info"]("r1", {})
    assert resp["id"] == "r1"
    assert resp["jsonrpc"] == "2.0"
    assert isinstance(resp["result"], dict)
    assert resp["result"].get("enabled") is False  # fail-open, no pet configured


def test_pet_changed_watcher_patch_liveness(monkeypatch, tmp_path):
    # test_change_watcher monkeypatches server._pet_active_selection; the
    # re-export binding must keep intercepting (consensus section 3 seam #3).
    sheet = _png(tmp_path / "sheet.png")
    fake = lambda: (True, _fake_pet(sheet, slug="patch-pet", display_name="Patch Pet"), 0.5)
    monkeypatch.setattr(server, "_pet_active_selection", fake)
    payload = server._pet_changed_payload()
    assert payload == {
        "enabled": True,
        "slug": "patch-pet",
        "displayName": "Patch Pet",
        "scale": 0.5,
        "spritesheetRevision": f"{sheet.stat().st_mtime_ns}:{sheet.stat().st_size}",
    }


def test_pet_sig_watcher_second_hop(monkeypatch, tmp_path):
    # _pet_sig (defined in server.py, R1) reads _pet_active_selection +
    # _pet_sheet_revision through the server namespace.
    sheet = _png(tmp_path / "sheet.png")
    monkeypatch.setattr(server, "_load_cfg", lambda: {"display": {"pet": {"enabled": True}}})
    monkeypatch.setattr(
        server, "_pet_active_selection",
        lambda: (True, _fake_pet(sheet, slug="sig-pet"), 0.75),
    )
    assert server._pet_sig() == (
        "sig-pet",
        f"{sheet.stat().st_mtime_ns}:{sheet.stat().st_size}",
        0.75,
    )


# ---------------------------------------------------------------------------
# Payload shape, cache, clone-on-read (wire contract, consensus section 2.5)
# ---------------------------------------------------------------------------

def test_sprite_payload_wire_shape(tmp_path):
    sheet = _png(tmp_path / "sheet.png")
    payload = pet_payload._pet_sprite_payload(_fake_pet(sheet), scale=1.0)
    assert set(payload) == WIRE_KEYS
    assert payload["slug"] == "test-pet"
    assert payload["displayName"] == "Test Pet"
    assert payload["mime"] == "image/png"
    assert base64.b64decode(payload["spritesheetBase64"]) == sheet.read_bytes()
    assert payload["scale"] == 1.0
    assert isinstance(payload["framesByState"], dict)
    assert isinstance(payload["stateRows"], list)


def test_cache_hit_and_clone_on_read_no_alias(tmp_path):
    sheet = _png(tmp_path / "sheet.png")
    pet = _fake_pet(sheet)
    first = pet_payload._pet_sprite_payload(pet, scale=1.0)
    assert len(pet_payload._pet_payload_cache) == 1

    # Mutating the returned clone must not corrupt the cache entry.
    first["slug"] = "MUTATED"
    first["framesByState"]["hacked"] = 999
    cached = next(iter(pet_payload._pet_payload_cache.values()))
    assert cached["slug"] == "test-pet"
    assert "hacked" not in cached["framesByState"]

    # Second call: distinct object, same wire values (served from cache).
    second = pet_payload._pet_sprite_payload(pet, scale=1.0)
    assert second is not first
    assert second["slug"] == "test-pet"
    assert second["spritesheetBase64"] == base64.standard_b64encode(
        sheet.read_bytes()
    ).decode("ascii")
    assert len(pet_payload._pet_payload_cache) == 1


def test_cache_key_change_busts_cache(tmp_path):
    sheet = _png(tmp_path / "sheet.png")
    pet = _fake_pet(sheet)
    pet_payload._pet_sprite_payload(pet, scale=1.0)
    assert len(pet_payload._pet_payload_cache) == 1
    # mtime bump -> new key -> new entry (cap still enforced)
    time.sleep(0.02)
    os_utime_backdate = time.time() + 5  # future mtime, definitely different
    import os
    os.utime(sheet, (os_utime_backdate, os_utime_backdate))
    pet_payload._pet_sprite_payload(pet, scale=1.0)
    assert len(pet_payload._pet_payload_cache) == 2


def test_cache_cap_stays_at_8(tmp_path):
    for i in range(10):
        sheet = _png(tmp_path / f"sheet{i}.png", size=(32 + i, 32 + i))
        pet_payload._pet_sprite_payload(_fake_pet(sheet, slug=f"pet-{i}"), scale=1.0)
    assert len(pet_payload._pet_payload_cache) <= 8


def test_sprite_payload_scale_is_part_of_key(tmp_path):
    sheet = _png(tmp_path / "sheet.png")
    pet = _fake_pet(sheet)
    pet_payload._pet_sprite_payload(pet, scale=1.0)
    pet_payload._pet_sprite_payload(pet, scale=2.0)
    assert len(pet_payload._pet_payload_cache) == 2


# ---------------------------------------------------------------------------
# Cache key / revision helpers
# ---------------------------------------------------------------------------

def test_payload_cache_key_shape(tmp_path):
    sheet = _png(tmp_path / "sheet.png")
    key = pet_payload._pet_payload_cache_key(_fake_pet(sheet, slug="k", display_name="K"), scale=1.25)
    assert key == (
        str(sheet), sheet.stat().st_mtime_ns, sheet.stat().st_size,
        "k", "K", round(1.25, 4),
    )


def test_payload_cache_key_missing_file_returns_none(tmp_path):
    missing = tmp_path / "nope.png"
    assert pet_payload._pet_payload_cache_key(_fake_pet(missing), scale=1.0) is None


def test_sheet_revision_shape(tmp_path):
    sheet = _png(tmp_path / "sheet.png")
    assert pet_payload._pet_sheet_revision(sheet) == f"{sheet.stat().st_mtime_ns}:{sheet.stat().st_size}"


def test_sheet_revision_fail_open(tmp_path):
    assert pet_payload._pet_sheet_revision(tmp_path / "missing.png") == "0:0"


# ---------------------------------------------------------------------------
# Reference-image data URL validation
# ---------------------------------------------------------------------------

def _data_url(mime, raw):
    return f"data:image/{mime};base64," + base64.b64encode(raw).decode("ascii")


def test_reference_images_valid_png(tmp_path):
    raw = _png(tmp_path / "src.png").read_bytes()
    out = pet_payload._pet_reference_images_from_data_url(_data_url("png", raw), tmp_path)
    assert out == [tmp_path / "reference.png"]
    assert (tmp_path / "reference.png").read_bytes() == raw


def test_reference_images_valid_jpeg(tmp_path):
    raw = b"\xff\xd8\xff\xe0fakejpeg"
    out = pet_payload._pet_reference_images_from_data_url(_data_url("jpeg", raw), tmp_path)
    assert out == [tmp_path / "reference.jpg"]


def test_reference_images_mime_whitelist(tmp_path):
    with pytest.raises(ValueError, match="unsupported reference image type"):
        pet_payload._pet_reference_images_from_data_url(_data_url("bmp", b"x" * 16), tmp_path)


def test_reference_images_invalid_format(tmp_path):
    with pytest.raises(ValueError, match="invalid reference image format"):
        pet_payload._pet_reference_images_from_data_url("not-a-data-url", tmp_path)


def test_reference_images_size_cap(monkeypatch, tmp_path):
    monkeypatch.setattr(pet_payload, "_PET_REFERENCE_MAX_BYTES", 10)
    with pytest.raises(ValueError, match="reference image too large"):
        pet_payload._pet_reference_images_from_data_url(_data_url("png", b"x" * 64), tmp_path)


def test_reference_images_invalid_base64(tmp_path):
    with pytest.raises(ValueError, match="invalid reference image data"):
        pet_payload._pet_reference_images_from_data_url(
            "data:image/png;base64,!!!not-base64!!!", tmp_path
        )


# ---------------------------------------------------------------------------
# Cancel token lifecycle
# ---------------------------------------------------------------------------

def test_cancel_lifecycle():
    assert pet_payload._pet_is_cancelled("tok") is False
    pet_payload._pet_cancel_request("tok")
    assert pet_payload._pet_is_cancelled("tok") is True
    pet_payload._pet_cancel_arm("tok")
    assert pet_payload._pet_is_cancelled("tok") is False
    pet_payload._pet_cancel_request("tok")
    pet_payload._pet_cancel_release("tok")
    assert pet_payload._pet_is_cancelled("tok") is False


def test_cancel_tokens_isolated():
    pet_payload._pet_cancel_request("a")
    assert pet_payload._pet_is_cancelled("a") is True
    assert pet_payload._pet_is_cancelled("b") is False
    pet_payload._pet_cancel_release("b")  # release of unset token is a no-op
    assert pet_payload._pet_is_cancelled("a") is True


# ---------------------------------------------------------------------------
# PNG data-URI downscale
# ---------------------------------------------------------------------------

def test_png_data_uri_downscales(tmp_path):
    src = _png(tmp_path / "big.png", size=(400, 400))
    uri = pet_payload._pet_png_data_uri(src, max_px=64)
    assert uri.startswith("data:image/png;base64,")
    decoded = base64.b64decode(uri.split(",", 1)[1])
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"
    with Image.open(__import__("io").BytesIO(decoded)) as img:
        assert max(img.size) <= 64


# ---------------------------------------------------------------------------
# Gen sweep staleness
# ---------------------------------------------------------------------------

def test_gen_sweep_removes_only_stale(tmp_path):
    root = tmp_path / "pet-gen"
    root.mkdir()
    old = root / "old"
    fresh = root / "fresh"
    old.mkdir()
    fresh.mkdir()
    backdate = time.time() - 7200
    import os
    os.utime(old, (backdate, backdate))
    pet_payload._pet_gen_sweep(root, max_age_s=3600.0)
    assert not old.exists()
    assert fresh.exists()


def test_gen_sweep_missing_root_no_raise(tmp_path):
    pet_payload._pet_gen_sweep(tmp_path / "missing", max_age_s=1.0)  # no exception


# ---------------------------------------------------------------------------
# Config scale fail-open
# ---------------------------------------------------------------------------

def test_config_scale_reads_display_pet_scale(monkeypatch):
    import hermes_cli.config
    monkeypatch.setattr(
        hermes_cli.config, "load_config",
        lambda: {"display": {"pet": {"scale": 2.5}}},
    )
    assert pet_payload._pet_config_scale() == 2.5


def test_config_scale_fail_open(monkeypatch):
    import hermes_cli.config
    from agent.pet import constants

    def boom():
        raise RuntimeError("config broken")

    monkeypatch.setattr(hermes_cli.config, "load_config", boom)
    assert pet_payload._pet_config_scale() == constants.DEFAULT_SCALE


def test_config_scale_missing_keys_fail_open(monkeypatch):
    import hermes_cli.config
    from agent.pet import constants
    monkeypatch.setattr(hermes_cli.config, "load_config", lambda: {})
    assert pet_payload._pet_config_scale() == constants.DEFAULT_SCALE


# ---------------------------------------------------------------------------
# Fail-open decode helpers
# ---------------------------------------------------------------------------

def test_frame_counts_fail_open_on_garbage(tmp_path):
    garbage = tmp_path / "sheet.png"
    garbage.write_bytes(b"not an image")
    # Fail-open contract: decode hiccups degrade to a dict (never raise).
    counts = pet_payload._pet_frame_counts(garbage)
    assert isinstance(counts, dict)
    rows = pet_payload._pet_row_frame_counts(garbage)
    assert isinstance(rows, dict)
    states = pet_payload._pet_state_rows(garbage)
    assert isinstance(states, list)
