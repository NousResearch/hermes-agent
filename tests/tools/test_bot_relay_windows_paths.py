"""Windows-path viability and venv CLI resolution for bot relay (#93590).

Two failures on a Windows desktop install talking to a remote gateway:

1. ``waiter_command`` embeds the reply path into generated ``python -c``
   source with ``!r``. repr escapes each backslash, but the Windows
   execution layer the waiter runs under folds ``\\`` back to ``\`` —
   ``\\U`` in ``C:\\Users\\...`` then parses as a unicode escape and
   SyntaxErrors the whole script. The raw-string prefix keeps the folded
   single backslash a literal; POSIX paths contain no backslashes, so it
   is a no-op there, and ``\\'`` inside a raw literal still cannot
   terminate the string, so the injection defense from #93091's
   python -c hardening is unchanged.

2. ``local_delivery_command`` hardcoded ``"hermes"``, relying on PATH —
   which service contexts (systemd units, desktop launchers, non-login
   SSH shells) do not provide, so delivery died with ENOENT. It now
   resolves the CLI next to this gateway's own interpreter (the venv
   bin/Scripts sibling), falling back to the bare name. The #93091
   turn-lock recognition in bot_mode_dm matches the CLI element by
   basename so resolved absolute paths (and ``hermes.exe``) still take
   the per-profile lock.
"""

import ast
import shlex
from pathlib import Path

import tools.bot_mode_dm as bot_mode_dm
import tools.bot_relay as bot_relay


ENV = {"id": "d" * 32, "target_handle": "researcher", "target_connection": "ssh-vps"}


def _waiter_code(root, env=None) -> str:
    cmd = bot_relay.waiter_command(root, env or ENV)
    parts = shlex.split(cmd)
    return parts[parts.index("-c") + 1]


def test_waiter_windows_path_compiles_after_backslash_folding():
    """A Windows reply path must survive the execution layer folding the
    repr-escaped double backslash back to a single one — the exact shape
    that SyntaxErrored with ``\\U`` on #93590's reporter setup."""
    code = _waiter_code("C:\\Users\\joshu\\.hermes")
    assert "C:" in code  # sanity: the Windows path made it into the payload
    folded = code.replace("\\\\", "\\")
    # Raw literals: `p = r'C:\Users\joshu\...'` — no unicode-escape crash.
    compile(folded, "<waiter>", "exec")


def test_waiter_posix_path_and_label_values_roundtrip():
    """On POSIX (backslash-free paths) the raw prefix changes nothing."""
    root = Path("/tmp/hermes-home")
    code = _waiter_code(root)
    assigns = {
        t.targets[0].id: t.value
        for t in ast.parse(code).body
        if isinstance(t, ast.Assign) and isinstance(t.targets[0], ast.Name)
    }
    expected = str(root / "bot_relay" / "replies" / f"{ENV['id']}.json")
    assert assigns["p"].value == expected
    assert assigns["label"].value == "@researcher on ssh-vps"
    # The literals are raw-prefixed in the generated source.
    assert "\np = r'" in code
    assert "\nlabel = r'" in code


def test_waiter_raw_prefix_keeps_injection_defense():
    """Hostile roster fields must stay data under the raw prefix too."""
    inj = {
        "id": "e" * 32,
        "target_handle": "researcher",
        "target_connection": "x'); __import__('sys').exit(2); print('x",
    }
    code = _waiter_code(Path("/tmp/hermes-home"), inj)
    compile(code, "<waiter>", "exec")
    calls = [
        n.func.id
        for n in ast.walk(ast.parse(code))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    ]
    # The generated waiter only calls str/print/compile-free builtins by
    # name; the payload's __import__ must remain a string literal, not a
    # live call — parse it back and confirm it stayed data.
    assert "__import__" not in calls
    assert "x'); __import__('sys').exit(2); print('x" in code


def test_local_delivery_resolves_sibling_hermes(tmp_path, monkeypatch):
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    sibling = bin_dir / "hermes"
    sibling.touch()
    sibling.chmod(0o755)
    monkeypatch.setattr("sys.executable", str(bin_dir / "python"))

    argv = bot_relay.local_delivery_command("ops", "query.json")
    assert argv[0] == str(sibling)
    assert argv[1:3] == ["-p", "ops"]
    assert argv[argv.index("--query-file") + 1] == "query.json"


def test_local_delivery_falls_back_to_bare_name(tmp_path, monkeypatch):
    empty = tmp_path / "nowhere"
    empty.mkdir(parents=True)
    monkeypatch.setattr("sys.executable", str(empty / "python"))

    argv = bot_relay.local_delivery_command("ops", "query.json")
    assert argv[0] == "hermes"
    assert argv[1:3] == ["-p", "ops"]


def test_delivery_lock_recognizes_resolved_cli_paths(tmp_path, monkeypatch):
    """The #93091 per-profile turn lock must keep matching delivery argvs
    now that argv[0] may be a resolved absolute path (or hermes.exe)."""
    acquired = []

    class _Ctx:
        def __enter__(self):
            acquired.append("locked")
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(bot_relay, "acquire_turn_lock", lambda root, profile: _Ctx())
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    with bot_mode_dm._delivery_lock(
        [str(tmp_path / "venv" / "bin" / "hermes"), "-p", "ops", "chat"],
        stdin_file=False,
    ):
        pass
    with bot_mode_dm._delivery_lock(["hermes", "-p", "ops", "chat"], stdin_file=False):
        pass
    with bot_mode_dm._delivery_lock(
        ["C:\\venv\\Scripts\\hermes.exe", "-p", "ops", "chat"], stdin_file=False
    ):
        pass
    assert acquired == ["locked", "locked", "locked"]

    # Unrelated argvs still bypass the lock entirely.
    with bot_mode_dm._delivery_lock(["python", "-m", "whatever"], stdin_file=False):
        pass
    assert acquired == ["locked", "locked", "locked"]
