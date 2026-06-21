"""Contract test: the s6-overlay stage2 hook re-chowns the data-volume
hermes-owned subdirs under $HERMES_HOME to the runtime hermes UID whenever any
of them is not already hermes-owned — INDEPENDENTLY of whether the top-level
$HERMES_HOME ownership already matches.

Regression guard for the HERMES_UID/HERMES_GID/PUID/PGID remap path (#41699),
a sibling of the build-tree regression #38556 fixed for $INSTALL_DIR.

`usermod -u <new> hermes` re-chowns the hermes home dir ($HERMES_HOME ==
/opt/data) to the new UID as a side effect. The data-volume targeted chown was
gated behind `stat $HERMES_HOME != hermes_uid`, so after any remap that stat is
already satisfied and the subdir/state-file chown was silently skipped —
leaving subdirs owned by the build-time UID (10000) (case a) or with a stale
GID because `groupmod` does not re-chown files (case b). The fix probes the
hermes-owned subdirs directly rather than gating solely on $HERMES_HOME.

The extraction + stubbed-shell-run approach mirrors
tests/tools/test_stage2_hook_build_tree_chown.py.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE2_HOOK = REPO_ROOT / "docker" / "stage2-hook.sh"

# The canonical hermes-owned subdir list the seed/chown block manages.
SUBDIRS = [
    "cron",
    "sessions",
    "logs",
    "hooks",
    "memories",
    "skills",
    "skins",
    "plans",
    "workspace",
    "home",
    "profiles",
    "pairing",
    "platforms/pairing",
]


@pytest.fixture(scope="module")
def stage2_text() -> str:
    if not STAGE2_HOOK.exists():
        pytest.skip("docker/stage2-hook.sh not present in this checkout")
    return STAGE2_HOOK.read_text()


def _data_volume_block(text: str) -> str:
    """Extract the data-volume chown block: from the `needs_chown=false`
    gate through the closing `fi` of the `if [ "$needs_chown" = true ]` body."""
    # Capture from the `needs_chown` gate through the closing `fi` of the
    # `if [ "$needs_chown" = true ]` body. Anchor on the body's recursive
    # subdir chown so the match spans the WHOLE block (gate + body), not just
    # the probe loop's earlier `done`/`fi`.
    m = re.search(
        r"(actual_hermes_uid=\$\(id -u hermes\)[\s\S]*?"
        r'chown -R hermes:hermes "\$HERMES_HOME/\$sub"[\s\S]*?\n    done\nfi)',
        text,
    )
    assert m, (
        "stage2-hook.sh must contain the needs_chown-gated data-volume chown "
        "block ending in the subdir recursive-chown `done` + `fi`"
    )
    return m.group(1)


def test_data_volume_chown_probes_subdirs(stage2_text: str) -> None:
    """The needs_chown gate must probe the hermes-owned subdirs directly, not
    rely solely on the top-level $HERMES_HOME ownership check — that solely
    gated form is exactly the #41699 (#35027/#38556 family) regression."""
    block = _data_volume_block(stage2_text)
    # A per-subdir ownership probe must exist (not only the $HERMES_HOME stat).
    # It must compare BOTH uid and gid (%u:%g) so a GID-only remap is caught.
    assert re.search(
        r'stat -c %u:%g "\$HERMES_HOME/\$sub"', block
    ), "data-volume needs_chown must probe each subdir's uid:gid directly"
    # The probe must loop over the canonical hermes-owned subdir list.
    for sub in ("cron", "sessions", "platforms/pairing"):
        assert sub in block, f"subdir probe/chown must cover {sub}"


def _run_data_volume_block(
    text: str,
    *,
    hermes_owner: str,
    home_owner: str,
    subdir_owner: str,
) -> bool:
    """Run the extracted data-volume block against a real tmp $HERMES_HOME with
    the subdirs present, with `stat` and `chown` stubbed. Owners are
    ``"uid:gid"`` strings. `stat -c %u:%g <path>` returns ``home_owner`` for the
    top-level dir and ``subdir_owner`` for any subdir; the runtime hermes
    identity is ``hermes_owner``. Returns True iff the block attempted a chown."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    block = _data_volume_block(text)
    hermes_uid, hermes_gid = hermes_owner.split(":")

    with tempfile.TemporaryDirectory() as d:
        dpath = Path(d)
        home = dpath / "data"
        for sub in SUBDIRS:
            (home / sub).mkdir(parents=True, exist_ok=True)
        log = dpath / "chown.log"
        # Stubs:
        #   stat -c <fmt> <path> -> owner for path; honours the FORMAT arg so a
        #       `%u` (uid-only) probe and a `%u:%g` (uid:gid) probe return
        #       different strings. This is what lets the GID-only test detect a
        #       uid-only gate: under `%u` the stale GID is invisible.
        #   chown ...            -> record that it fired
        # The real $HERMES_HOME dir tree backs the `[ -e ]` guards.
        script = (
            "set -eu\n"
            f'HERMES_HOME="{home}"\n'
            "id() { :; }\n"  # never invoked; actual_hermes_uid/gid set below
            f"actual_hermes_uid={hermes_uid}\n"
            f"actual_hermes_gid={hermes_gid}\n"
            "stat() {\n"
            '    fmt="$2"; target="$3"\n'
            f'    if [ "$target" = "{home}" ]; then owner="{home_owner}";\n'
            f'    else owner="{subdir_owner}"; fi\n'
            '    if [ "$fmt" = "%u" ]; then echo "${owner%%:*}";\n'
            '    else echo "$owner"; fi\n'
            "}\n"
            f'chown() {{ echo fired >> "{log}"; }}\n'
            + block
        )
        # The extracted block opens with `actual_hermes_uid=$(id -u hermes)` and
        # `actual_hermes_gid=$(id -g hermes)`, which would override our stubs —
        # re-pin both right at the block's own assignments.
        script = script.replace(
            "actual_hermes_uid=$(id -u hermes)",
            f"actual_hermes_uid={hermes_uid}",
        ).replace(
            "actual_hermes_gid=$(id -g hermes)",
            f"actual_hermes_gid={hermes_gid}",
        )
        script_path = dpath / "harness.sh"
        script_path.write_text(script)
        proc = subprocess.run([bash, str(script_path)], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        return log.exists() and "fired" in log.read_text()


def test_chown_fires_when_subdir_owner_differs(stage2_text: str) -> None:
    """#41699 repro: after a HERMES_UID/PUID remap `usermod` already chowned
    $HERMES_HOME to the new UID (home_owner == hermes_owner), but the subdirs
    are still owned by the build-time UID (10000). The targeted chown MUST
    fire."""
    fired = _run_data_volume_block(
        stage2_text, hermes_owner="4242:4242", home_owner="4242:4242", subdir_owner="10000:10000"
    )
    assert fired, (
        "data-volume chown must fire when a hermes-owned subdir is not owned by "
        "the runtime hermes UID, even though $HERMES_HOME already matches "
        "(#41699 regression)"
    )


def test_chown_fires_when_hermes_home_owner_differs(stage2_text: str) -> None:
    """Original #35027 path intact: $HERMES_HOME itself owned by the build UID
    (10000) while hermes is now 4242 — the chown must still fire."""
    fired = _run_data_volume_block(
        stage2_text, hermes_owner="4242:4242", home_owner="10000:10000", subdir_owner="4242:4242"
    )
    assert fired, (
        "data-volume chown must still fire on the original $HERMES_HOME-owner "
        "mismatch path"
    )


def test_chown_fires_on_gid_only_remap(stage2_text: str) -> None:
    """#41699 GID-only repro: a `groupmod -o -g <new> hermes` PGID remap leaves
    every UID unchanged (usermod/groupmod do NOT re-chown files), so the UID of
    $HERMES_HOME and the subdirs still matches hermes — only the GID is stale.
    A uid-only gate (`stat -c %u`) would miss this entirely and skip the chown,
    leaving stale group ownership. Comparing %u:%g MUST fire the chown."""
    fired = _run_data_volume_block(
        stage2_text, hermes_owner="10000:5555", home_owner="10000:10000", subdir_owner="10000:10000"
    )
    assert fired, (
        "data-volume chown must fire on a GID-only remap (uid matches but gid is "
        "stale) — a uid-only gate would skip it and leave stale group ownership "
        "(#41699)"
    )


def test_chown_skipped_when_all_owned(stage2_text: str) -> None:
    """Idempotency / negative: once $HERMES_HOME and every subdir are
    hermes-owned, the expensive recursive chown is skipped on subsequent
    boots."""
    fired = _run_data_volume_block(
        stage2_text, hermes_owner="4242:4242", home_owner="4242:4242", subdir_owner="4242:4242"
    )
    assert not fired, (
        "data-volume chown must be skipped when $HERMES_HOME and all subdirs "
        "already match the runtime hermes UID (avoid expensive recursive chown "
        "on every restart)"
    )


def test_chown_skipped_for_default_uid(stage2_text: str) -> None:
    """No remap: home + subdirs owned by the default build UID (10000) and
    hermes is still 10000 — nothing to do."""
    fired = _run_data_volume_block(
        stage2_text, hermes_owner="10000:10000", home_owner="10000:10000", subdir_owner="10000:10000"
    )
    assert not fired
