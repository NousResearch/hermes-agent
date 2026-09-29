"""Security review (t_02ee0bb4 / t_ad082a6c): terminal-guard bypass shapes.

The approvals guard classified the COMMAND STRING only, so every spelling that
hid the protected path behind an environment variable, a staging variable, a
`cd`, or a `$(...)` substitution slipped through auto-approve while the literal
spelling of the same file was flagged. These pin:

* the six confirmed bypass shapes (R1 env-value fold + R1b non-literal operand),
* root and sibling `config.yaml` writes,
* `.env.example` — the file a blocked read is TOLD to use — staying usable,
* the controls that must NOT flip (literal reads, non-Hermes paths, prose).
"""
from __future__ import annotations

import os

import pytest

from tools.approval_detection import detect_dangerous_command

ROOT = os.path.join(os.path.expanduser("~"), "AppData", "Local", "hermes")
OTHER = "other-profile"


@pytest.fixture(autouse=True)
def _localappdata(monkeypatch):
    """Pin LOCALAPPDATA to a home-relative path so the fold is deterministic on
    every platform (POSIX CI has no LOCALAPPDATA, and a temp dir off-home would
    not survive the user-home fold)."""
    monkeypatch.setenv("LOCALAPPDATA", os.path.join(os.path.expanduser("~"), "AppData", "Local"))


def _flagged(command: str) -> bool:
    return bool(detect_dangerous_command(command)[0])


# --- the six bypass shapes -------------------------------------------------


def test_var_indirection_direct():
    """Shape 1: `$LOCALAPPDATA/...` — the literal path was flagged, the `$VAR`
    spelling was an opaque token no path rule could match."""
    assert _flagged(f'cat "$LOCALAPPDATA/hermes/profiles/{OTHER}/.env"')


def test_staging_variable():
    """Shape 2: a shell variable assigned from `$LOCALAPPDATA`, then used with
    `$ROOT`. Nothing in the second half of the command names the Hermes tree."""
    assert _flagged(
        f'ROOT="$LOCALAPPDATA/hermes/profiles"; cat "$ROOT/{OTHER}/.env"'
    )
    assert _flagged(
        f"ROOT=\"$LOCALAPPDATA/hermes/profiles\"; sed -i 's/X/Y/' \"$ROOT/{OTHER}/.env\""
    )


def test_cd_then_relative_operand():
    """Shape 3: `cd` into the directory, then a BARE basename. Positionless
    rule 266 still sees the resolved directory and the basename in one command."""
    assert _flagged(
        f'cd "$LOCALAPPDATA/hermes/profiles/{OTHER}" && sed -i \'s/X/Y/\' .env'
    )
    assert _flagged(f'cd "$LOCALAPPDATA/hermes/profiles/{OTHER}" && cat .env')


def test_percent_expansion_cmd_form():
    """Shape 4: `%LOCALAPPDATA%` — the cmd.exe spelling of the same variable."""
    assert _flagged(
        f'type "%LOCALAPPDATA%\\hermes\\profiles\\{OTHER}\\.env"'
    )


def test_substitution_path():
    """Shape 6: `$(...)` builds the directory at run time, so the operand is
    non-literal and no path rule can ever see through it (R1b)."""
    assert _flagged(f'cat "$(dirname /x/y)/{OTHER}/.env"')
    assert _flagged(f'cat "$(printf \'%s\' \'{ROOT}/profiles/{OTHER}/.env\')')


def test_bare_basename_read_is_a_documented_residual(tmp_path):
    """Shape 5 (bare `.env` with the cwd already inside the profile).

    The terminal layer classifies ONE command string and has no cwd, so a bare
    basename cannot be attributed to the profile — and the suite pins the
    mirror case as safe (`cat .env > backup.txt` in
    ``test_adjacent_filenames_stay_safe``). The compensating control is the
    file-tool read deny, which resolves the path and blocks it on every
    platform. Pinned here so the residual is visible rather than assumed.
    """
    assert not _flagged("cat .env")
    from agent.file_safety import get_read_block_error
    # Any path: the read deny is basename-based, so the compensating control is
    # exercised without touching the real home (tests/home_io_guard.py forbids it).
    assert get_read_block_error(os.path.join(str(tmp_path), "profiles", OTHER, ".env"))


# --- config.yaml: root and sibling -----------------------------------------


def test_root_config_yaml_write_is_flagged():
    """The ROOT config.yaml is the approval policy (approvals.mode / yolo) and
    the cache is mtime-keyed, but only `~/.hermes/config.yaml` — the ACTIVE
    profile's spelling — was ever matched."""
    assert _flagged(f"sed -i 's/a/b/' \"{ROOT}/config.yaml\"")
    assert _flagged(f'R="$LOCALAPPDATA/hermes"; sed -i \'s/a/b/\' "$R/config.yaml"')


def test_sibling_profile_config_yaml_write_is_flagged():
    assert _flagged(f"sed -i 's/a/b/' \"{ROOT}/profiles/{OTHER}/config.yaml\"")
    assert _flagged(
        f'R="$LOCALAPPDATA/hermes/profiles"; sed -i \'s/a/b/\' "$R/{OTHER}/config.yaml"'
    )


def test_non_literal_operand_to_a_protected_store():
    """R1b: the operand is BUILT from an expansion, so the path rules can never
    match it — only the shape of the word is visible."""
    assert _flagged(f'R="$LOCALAPPDATA/hermes"; cat "$R/auth.json"')
    assert _flagged(f'R="$LOCALAPPDATA/hermes/profiles"; sed -i \'s/a/b/\' "$R/{OTHER}/SOUL.md"')


# --- .env.example stays usable ---------------------------------------------


def test_env_example_reads_are_not_flagged():
    """`.env.example` is the documented substitute a blocked `.env` read points
    at; rule 266 used to flag it anyway, closing the loop on its own remediation."""
    for command in (
        f'cat "{ROOT}/.env.example"',
        f'cat "{ROOT}/profiles/{OTHER}/.env.example"',
        "cat $HOME/.hermes/.env.example",
    ):
        assert not _flagged(command), command


def test_env_example_write_is_not_flagged():
    for command in ("echo X=1 > .env.example", "cp /tmp/t .env.example"):
        assert not _flagged(command), command


# --- controls that must NOT flip -------------------------------------------


def test_literal_reads_and_non_hermes_paths_stay_safe():
    for command in (
        "cat ~/.hermes/config.yaml",
        "sed -i 's/a/b/' /srv/app/config.yaml",
        "cp config.yaml backup.yaml",
        "cat .env > backup.txt",
        "echo x > .env#backup",
        "echo x > config.yaml.bak",
        "grep -rn 'appdata/local/hermes' website/docs",
        "cat ./readme.txt",
    ):
        assert not _flagged(command), command
