from pathlib import Path


ENTRYPOINT = Path(__file__).resolve().parents[2] / "docker" / "entrypoint.sh"


def test_entrypoint_supports_puid_pgid_aliases_and_prefers_hermes_uid_gid() -> None:
    """`PUID`/`PGID` should only be used when `HERMES_UID`/`HERMES_GID` are unset."""

    text = ENTRYPOINT.read_text(encoding="utf-8")

    assert 'if [ -z "$HERMES_UID" ] && [ -n "$PUID" ]; then' in text
    assert '    HERMES_UID="$PUID"' in text
    assert 'if [ -z "$HERMES_GID" ] && [ -n "$PGID" ]; then' in text
    assert '    HERMES_GID="$PGID"' in text
