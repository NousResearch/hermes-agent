"""At-rest permissions for the media caches created by ``tools.vision_tools``.

Scope: this file guards **``tools/vision_tools.py``** — its
``_secure_cache_dir`` / ``_write_private_bytes`` / ``_precreate_private_file``
helpers and the four call sites that use them. The sibling suite
``tests/computer_use/test_computer_use_vision_cache_permissions.py`` guards the
*other* creator of ``cache/vision``, ``tools/computer_use/tool.py``. Both are
kept because either module can win the race to create that shared directory;
``test_agrees_with_computer_use_cache_hardening`` below is the seam that pins
them together.

``$HERMES_HOME/cache/vision`` and ``cache/video`` hold whatever the agent was
asked to look at — a private attachment, an internal screenshot, a document
scan. Created with a bare ``mkdir(parents=True, exist_ok=True)`` they inherited
the umask and landed 0755, and the temp files inside landed 0644.

``HERMES_HOME`` is 0700 by default, so default-config exposure is narrow and
this is defence in depth. The concrete scenario is the documented
``HERMES_HOME_MODE=0701`` hatch (letting nginx/caddy traverse HERMES_HOME to
reach a served subdirectory), where a 0755 child really is world-readable.

POSIX-only: mode bits are advisory on Windows, where at-rest protection is
ACL-based and ``os.chmod`` only toggles the read-only flag.
"""

import os
import stat
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="POSIX mode bits; Windows at-rest protection is ACL-based",
)

GROUP_OTHER_BITS = (
    stat.S_IRGRP
    | stat.S_IWGRP
    | stat.S_IXGRP
    | stat.S_IROTH
    | stat.S_IWOTH
    | stat.S_IXOTH
)


def _mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


@pytest.fixture
def permissive_umask():
    """Force a world-readable-by-default umask for the duration of a test."""
    previous = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(previous)


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Never let the ambient environment silently disable the hardening.
    # tests/conftest.py blanks HERMES_CONTAINER / HERMES_HOME_MODE /
    # HERMES_MANAGED, but NOT HERMES_SKIP_CHMOD — so this is load-bearing,
    # not belt-and-braces.
    monkeypatch.delenv("HERMES_CONTAINER", raising=False)
    monkeypatch.delenv("HERMES_SKIP_CHMOD", raising=False)
    monkeypatch.delenv("HERMES_HOME_MODE", raising=False)
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    return home


@pytest.fixture
def neutralized_secure_dir(monkeypatch):
    """Make the post-creation reconciliation step a no-op.

    ``_secure_cache_dir`` hardens twice over: ``mode=`` on the ``mkdir`` call,
    then ``hermes_cli.config._secure_dir`` to reconcile policy. Either
    mechanism alone satisfies a plain "is it 0700?" assertion, so the two mask
    each other — with the reconciler in play every mode assertion in this file
    stays green even if ``mode=`` is deleted.

    Stubbing the reconciler out isolates the creation mode as the *only* thing
    that can produce the observed bits. Patched on ``hermes_cli.config``
    rather than on the caller because the production code imports it lazily
    inside the function, so the attribute is resolved at call time.
    """
    import hermes_cli.config as hermes_config

    calls = []

    def _recording_noop(path):
        calls.append(str(path))

    monkeypatch.setattr(hermes_config, "_secure_dir", _recording_noop)
    return calls


@pytest.fixture
def managed_nixos_home(hermes_home, monkeypatch):
    """Simulate a managed NixOS install that shares state via the hermes group.

    Reproduces the two conditions ``nix/nixosModules.nix`` actually creates:

    * ``systemd.tmpfiles`` pins ``stateDir/.hermes`` to ``2770`` — setgid,
      group-rwx — so the gateway and interactive ``hostUsers`` share it;
    * the service runs with ``UMask = "0007"``, commented "files created by
      the gateway should be group-writable so interactive users in the hermes
      group can read/write them".

    ``cache/vision`` and ``cache/video`` are *not* in those tmpfiles rules, so
    they are created lazily at runtime under exactly this umask — which is why
    the creation mode, not just the reconciliation, has to honour the carve-out.
    """
    monkeypatch.setenv("HERMES_MANAGED", "nixos")
    os.chmod(hermes_home, 0o2770)
    previous = os.umask(0o007)
    try:
        yield hermes_home
    finally:
        os.umask(previous)


@pytest.mark.parametrize(
    "new_subpath,old_name",
    [("cache/vision", "temp_vision_images"), ("cache/video", "temp_video_files")],
)
def test_fresh_media_cache_dir_has_no_group_or_other_access(
    hermes_home, permissive_umask, new_subpath, old_name
):
    """Every media cache this module creates must be owner-only."""
    from tools.vision_tools import _secure_cache_dir

    cache_dir = _secure_cache_dir(new_subpath, old_name)

    assert cache_dir.is_dir()
    mode = _mode(cache_dir)
    assert not (
        mode & GROUP_OTHER_BITS
    ), f"media cache {cache_dir} is group/other-accessible: {oct(mode)}"


def test_media_cache_mode_is_not_umask_derived(hermes_home, permissive_umask):
    """The mode must come from the code, not the ambient umask."""
    from tools.vision_tools import _secure_cache_dir

    mode = _mode(_secure_cache_dir("cache/vision", "temp_vision_images"))

    assert mode & stat.S_IRWXU == stat.S_IRWXU, "owner must retain full access"
    assert mode & GROUP_OTHER_BITS == 0, f"umask-derived mode leaked: {oct(mode)}"


def test_bare_mkdir_would_violate_the_contract(hermes_home, permissive_umask):
    """Guard against a vacuous suite.

    This is the exact pre-fix call shape. If it stops producing group/other
    bits under ``umask 022`` then the fixture — not the fix — is what the
    other tests are measuring, and they prove nothing.
    """
    from hermes_constants import get_hermes_dir

    pre_fix = get_hermes_dir("cache/pre-fix-shape", "temp_pre_fix_shape")
    pre_fix.mkdir(parents=True, exist_ok=True)

    mode = _mode(pre_fix)
    assert mode & GROUP_OTHER_BITS, (
        "pre-fix call shape no longer leaks group/other bits under umask 022; "
        f"got {oct(mode)} — the permission tests here would be vacuous"
    )


def test_preexisting_world_readable_cache_dir_is_healed(hermes_home, permissive_umask):
    """An older Hermes left 0755 behind; the next call must tighten it.

    Safe to do retroactively here specifically because this is Hermes-private
    scratch the same user re-reads in the same call — there is no user-shared
    content to strand.
    """
    from tools.vision_tools import _secure_cache_dir

    legacy = hermes_home / "cache" / "vision"
    legacy.mkdir(parents=True)
    os.chmod(legacy, 0o755)
    assert _mode(legacy) & stat.S_IROTH, "fixture precondition: starts world-readable"

    cache_dir = _secure_cache_dir("cache/vision", "temp_vision_images")

    assert cache_dir == legacy
    assert not (
        _mode(cache_dir) & GROUP_OTHER_BITS
    ), f"pre-existing dir left exposed: {oct(_mode(cache_dir))}"


def test_downloaded_media_bytes_are_written_owner_only(hermes_home, permissive_umask):
    """The downloaded bytes themselves must not land 0644.

    ``tools.computer_use.tool`` already writes 0600 into this directory; one
    directory with two file-mode conventions is the inconsistency that rots.
    """
    from tools.vision_tools import _secure_cache_dir, _write_private_bytes

    target = _secure_cache_dir("cache/vision", "temp_vision_images") / "temp_image.img"
    _write_private_bytes(target, b"\x89PNG\r\n\x1a\n private attachment")

    assert target.read_bytes().endswith(b"private attachment")
    mode = _mode(target)
    assert not (
        mode & GROUP_OTHER_BITS
    ), f"cached media {target} is group/other-readable: {oct(mode)}"


def test_private_write_overwrites_and_stays_readable(hermes_home, permissive_umask):
    """A permission fix that breaks the read path is the wrong fix.

    The vision pipeline writes the temp file then reads it straight back
    (base64-embed / MIME sniff), and re-uses the same name shape on retry, so
    truncating rewrite plus owner read must both keep working.
    """
    from tools.vision_tools import _secure_cache_dir, _write_private_bytes

    target = _secure_cache_dir("cache/vision", "temp_vision_images") / "reused.img"
    _write_private_bytes(target, b"first-and-longer-payload")
    _write_private_bytes(target, b"second")

    assert target.read_bytes() == b"second", "O_TRUNC rewrite must not leave residue"
    assert _mode(target) & stat.S_IRUSR, "owner must still be able to read it back"


def test_real_conversion_path_creates_hardened_cache(
    hermes_home, permissive_umask, tmp_path
):
    """Drive a real caller, not just the helper.

    ``_normalize_to_supported_image`` is the network-free entry point that
    materialises a converted PNG into the cache. Exercising it proves the
    hardening is actually wired into the production path and that conversion
    still works afterwards.
    """
    pytest.importorskip("PIL")
    from PIL import Image

    from tools.vision_tools import _normalize_to_supported_image

    source = tmp_path / "input.bmp"
    Image.new("RGB", (4, 4), (10, 20, 30)).save(source, format="BMP")

    out_path, mime, error = _normalize_to_supported_image(source, "image/bmp")

    assert error is None, f"conversion broke: {error}"
    assert mime == "image/png"
    assert out_path is not None and out_path.exists(), "converted file must exist"
    # The feature still works: the result is a readable PNG.
    with Image.open(out_path) as converted:
        assert converted.size == (4, 4)

    cache_dir = hermes_home / "cache" / "vision"
    assert out_path.parent == cache_dir
    assert not (
        _mode(cache_dir) & GROUP_OTHER_BITS
    ), f"production path left cache exposed: {oct(_mode(cache_dir))}"


def test_agrees_with_computer_use_cache_hardening(hermes_home, permissive_umask):
    """Both creators of ``cache/vision`` must land the same mode.

    Either module can win the race to create this directory. If they disagree,
    whichever runs first decides whether the captures are exposed.
    """
    from tools.computer_use.tool import _vision_cache_dir
    from tools.vision_tools import _secure_cache_dir

    from_vision_tools = _secure_cache_dir("cache/vision", "temp_vision_images")
    mode_after_vision_tools = _mode(from_vision_tools)

    from_computer_use = _vision_cache_dir()

    assert from_computer_use == from_vision_tools, "same directory expected"
    assert mode_after_vision_tools == _mode(from_computer_use), (
        "the two creators of cache/vision disagree on mode: "
        f"vision_tools={oct(mode_after_vision_tools)} "
        f"computer_use={oct(_mode(from_computer_use))}"
    )


def test_downloaded_video_lands_owner_only(hermes_home, permissive_umask):
    """The real video download path must not leave the .mp4 at 0644.

    ``_download_video`` ends in ``destination.write_bytes()``, which derives a
    fresh file's mode from the umask. ``video_analyze_tool`` pre-creates the
    target 0600 so the write inherits that inode's mode instead. Only the HTTP
    client is faked here — the pre-create, the guard, and the real write all
    execute.
    """
    import asyncio

    from tools.vision_tools import (
        _download_video,
        _precreate_private_file,
        _secure_cache_dir,
    )

    target = _secure_cache_dir("cache/video", "temp_video_files") / "temp_video_x.mp4"

    class FakeResponse:
        url = "https://allowed.test/clip.mp4"
        headers = {"content-length": "9"}
        content = b"fake-mp4-"

        def raise_for_status(self):
            return None

    def fake_client(**_kwargs):
        client = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client.get = AsyncMock(return_value=FakeResponse())
        return client

    with (
        patch("tools.vision_tools.check_website_access", return_value=None),
        patch("tools.url_safety.create_ssrf_safe_async_client", side_effect=fake_client),
    ):
        _precreate_private_file(target)
        asyncio.run(_download_video("https://allowed.test/clip.mp4", target, max_retries=1))

    assert target.read_bytes() == b"fake-mp4-", "the download must still land"
    mode = _mode(target)
    assert not (
        mode & GROUP_OTHER_BITS
    ), f"downloaded video {target} is group/other-readable: {oct(mode)}"


def test_unprepared_write_bytes_would_violate_the_contract(
    hermes_home, permissive_umask
):
    """Vacuity guard for the video-file contract.

    Without the pre-create, ``write_bytes`` on a fresh path is the pre-fix
    shape. If that stops leaking group/other bits under ``umask 022``, the
    test above proves nothing.
    """
    from tools.vision_tools import _secure_cache_dir

    pre_fix = _secure_cache_dir("cache/video", "temp_video_files") / "pre-fix-shape.mp4"
    pre_fix.write_bytes(b"fake-mp4-")

    mode = _mode(pre_fix)
    assert mode & GROUP_OTHER_BITS, (
        "pre-fix write_bytes shape no longer leaks group/other bits under "
        f"umask 022; got {oct(mode)} — the video-mode test would be vacuous"
    )


def test_precreate_does_not_clobber_an_existing_file(hermes_home, permissive_umask):
    """The pre-create must be additive, never destructive.

    It runs on a uuid path that should not exist, but it is ``O_EXCL`` so that
    a collision (or a caller reusing a path) can never truncate real bytes.
    """
    from tools.vision_tools import _precreate_private_file, _secure_cache_dir

    existing = _secure_cache_dir("cache/video", "temp_video_files") / "existing.mp4"
    existing.write_bytes(b"do-not-lose-me")

    _precreate_private_file(existing)

    assert existing.read_bytes() == b"do-not-lose-me", "pre-create truncated a file"


def test_failed_download_leaves_no_precreated_stub(hermes_home, permissive_umask):
    """Pre-creating the target must not litter the cache when a download fails.

    Creating the file before the download introduces an empty stub that did not
    exist before this change, so the existing cleanup has to cover it. Drives
    the real ``video_analyze_tool`` error path.
    """
    import asyncio

    from tools.vision_tools import video_analyze_tool

    async def boom(*_args, **_kwargs):
        raise RuntimeError("download exploded")

    with (
        patch("tools.vision_tools._validate_image_url_async", AsyncMock(return_value=True)),
        patch("tools.vision_tools.check_website_access", return_value=None),
        patch("tools.vision_tools._download_video", side_effect=boom),
    ):
        asyncio.run(video_analyze_tool("https://allowed.test/clip.mp4", "describe"))

    cache_dir = hermes_home / "cache" / "video"
    leftovers = sorted(p.name for p in cache_dir.iterdir()) if cache_dir.exists() else []
    assert leftovers == [], f"failed download left a stub behind: {leftovers}"


def test_managed_mode_leaves_existing_permissions_alone(
    hermes_home, permissive_umask, monkeypatch
):
    """Managed/NixOS installs own their own modes; we must not fight them."""
    monkeypatch.setenv("HERMES_MANAGED", "nixos")
    legacy = hermes_home / "cache" / "vision"
    legacy.mkdir(parents=True)
    os.chmod(legacy, 0o750)

    from tools.vision_tools import _secure_cache_dir

    assert (
        _mode(_secure_cache_dir("cache/vision", "temp_vision_images")) == 0o750
    ), "managed-mode permissions overridden"


@pytest.mark.parametrize(
    "new_subpath,old_name",
    [("cache/vision", "temp_vision_images"), ("cache/video", "temp_video_files")],
)
def test_creation_mode_alone_yields_owner_only(
    hermes_home, neutralized_secure_dir, new_subpath, old_name
):
    """``mode=`` on the ``mkdir`` call must be doing real work by itself.

    Two mechanisms harden these directories and either one satisfies a bare
    "is it 0700?" check, so they mask each other: with ``_secure_dir`` in play
    every mode assertion in this file stays green even if ``mode=`` is deleted.
    Here the reconciler is stubbed to a no-op, so the only thing that can
    produce owner-only bits is the mode passed at creation.

    This is the TOCTOU guarantee made observable. The window ``mode=`` closes —
    the instant between ``mkdir`` and a follow-up ``chmod`` where the cache
    sits world-readable — cannot be asserted by racing creation, but "correct
    without any chmod at all" is exactly equivalent and is deterministic.

    Deliberately not stacked with ``permissive_umask`` so a 0700 result cannot
    be an accident of a restrictive ambient umask on the machine running the
    suite; the sibling test below forces the umask wide open instead.
    """
    from tools.vision_tools import _secure_cache_dir

    cache_dir = _secure_cache_dir(new_subpath, old_name)

    assert cache_dir.is_dir()
    assert neutralized_secure_dir == [
        str(cache_dir)
    ], "reconciler stub was not exercised; this test is not isolating creation"
    mode = _mode(cache_dir)
    assert mode & GROUP_OTHER_BITS == 0, (
        f"with the reconciler neutralized {cache_dir} is group/other-accessible "
        f"({oct(mode)}) — the mode= on mkdir is not being applied, so the "
        "mkdir->chmod TOCTOU window is open"
    )
    assert mode & stat.S_IRWXU == stat.S_IRWXU, "owner must retain full access"


def test_creation_mode_is_not_inherited_from_a_permissive_umask(
    hermes_home, permissive_umask, neutralized_secure_dir
):
    """Same isolation, with the umask forced wide open.

    Together with the test above this pins the claim precisely: 0700 comes from
    the ``mode=`` argument, not from the ambient umask and not from the
    reconciler. ``Path.mkdir`` masks its ``mode`` with the umask, so ``mode=``
    of 0700 under ``umask 022`` still yields 0700 — but a *missing* ``mode=``
    yields 0755 and fails here.
    """
    from tools.vision_tools import _secure_cache_dir

    mode = _mode(_secure_cache_dir("cache/vision", "temp_vision_images"))

    assert neutralized_secure_dir, "reconciler stub was not exercised"
    assert (
        mode & GROUP_OTHER_BITS == 0
    ), f"umask-derived mode leaked with the reconciler neutralized: {oct(mode)}"


@pytest.mark.parametrize(
    "new_subpath,old_name",
    [("cache/vision", "temp_vision_images"), ("cache/video", "temp_video_files")],
)
def test_managed_mode_fresh_cache_keeps_group_sharing(
    managed_nixos_home, new_subpath, old_name
):
    """A *newly created* cache on NixOS must stay group-shareable.

    The sibling test above covers a pre-existing directory, where
    ``_secure_dir``'s ``is_managed()`` early return is enough. This is the
    other managed path: the directory does not exist yet, so the mode passed
    at *creation* is the only thing that decides it — and neither
    ``cache/vision`` nor ``cache/video`` is among the directories
    ``nix/nixosModules.nix`` pre-creates via ``systemd.tmpfiles``, so on a
    managed host it is always this path.

    Forcing 0700 here would not merely be cosmetic. The module shares
    ``$HERMES_HOME`` between the gateway service and interactive ``hostUsers``
    through the hermes group (``2770`` + ``UMask = "0007"`` + a deliberate
    refusal to ``chown -R``, which strips setgid). A 0700 cache created by
    whichever side runs first makes vision fail with EACCES for the other.
    """
    from tools.vision_tools import _secure_cache_dir

    expected = managed_nixos_home / new_subpath
    assert not expected.exists(), "fixture precondition: dir must be absent"

    cache_dir = _secure_cache_dir(new_subpath, old_name)

    assert cache_dir.is_dir(), "managed mode must still create the cache dir"
    mode = _mode(cache_dir)
    assert mode & stat.S_IRWXG, (
        f"fresh managed-mode {cache_dir} dropped group access ({oct(mode)}); the "
        "NixOS module's hermes-group sharing (2770 + UMask=0007) is broken, so "
        "an interactive hostUsers CLI and the gateway can no longer share it"
    )


def test_managed_mode_fresh_cache_agrees_across_both_creators(managed_nixos_home):
    """The two creators of ``cache/vision`` must agree on managed mode too.

    ``test_agrees_with_computer_use_cache_hardening`` pins this for the default
    install. Managed mode is the case where they could silently diverge: if one
    module honours the carve-out at creation and the other does not, whichever
    wins the race decides whether the hermes group keeps access.
    """
    from tools.computer_use.tool import _vision_cache_dir
    from tools.vision_tools import _secure_cache_dir

    from_vision_tools = _secure_cache_dir("cache/vision", "temp_vision_images")
    mode_after_vision_tools = _mode(from_vision_tools)

    from_computer_use = _vision_cache_dir()

    assert from_computer_use == from_vision_tools, "same directory expected"
    assert mode_after_vision_tools == _mode(from_computer_use), (
        "the two creators of cache/vision disagree on managed-mode creation: "
        f"vision_tools={oct(mode_after_vision_tools)} "
        f"computer_use={oct(_mode(from_computer_use))}"
    )
    assert mode_after_vision_tools & stat.S_IRWXG, (
        "both creators dropped hermes-group access on a managed install: "
        f"{oct(mode_after_vision_tools)}"
    )


def test_home_mode_override_is_honored(hermes_home, permissive_umask, monkeypatch):
    """The documented web-server traversal hatch still applies."""
    monkeypatch.setenv("HERMES_HOME_MODE", "0701")

    from tools.vision_tools import _secure_cache_dir

    mode = _mode(_secure_cache_dir("cache/vision", "temp_vision_images"))

    assert mode == 0o701
    # Execute-only: traversal without a directory listing. Files stay 0600.
    assert not (mode & (stat.S_IRGRP | stat.S_IROTH)), "listing must stay closed"


def test_container_deployments_still_get_a_usable_directory(
    hermes_home, permissive_umask, monkeypatch
):
    """A container/volume-mount deployment must not crash or lose the dir.

    ``_secure_dir`` checks ``is_managed()`` only — unlike ``_secure_file`` it
    does **not** consult ``_is_container()``, so the cache is still tightened
    to 0700 here. That adds no new restriction: ``ensure_hermes_home`` already
    puts HERMES_HOME and every standard subdir at 0700 in a container, and
    multi-UID deployments are served by ``HERMES_UID``/``HERMES_GID`` or
    ``HERMES_HOME_MODE``, both of which ``_secure_dir`` honours. So this
    asserts the weaker, true contract: the directory exists and nothing raised.
    """
    monkeypatch.setenv("HERMES_CONTAINER", "1")
    monkeypatch.setenv("HERMES_SKIP_CHMOD", "1")

    from tools.vision_tools import _secure_cache_dir

    assert _secure_cache_dir("cache/vision", "temp_vision_images").is_dir()
