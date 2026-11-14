"""#109367: destructive shell verbs on the Hermes home or its protected stores reach the approval gate.

The file tools already refuse to write these stores (agent.file_safety); the terminal gate used to
auto-approve deleting, moving, truncating, or SQL-wiping them. Reads and copies OUT stay clean.
"""

import time

import pytest

from tools.approval_detection import detect_dangerous_command


@pytest.fixture
def home(monkeypatch, tmp_path):
    path = tmp_path / "data"
    monkeypatch.setenv("HERMES_HOME", str(path))
    return str(path)


_DESTRUCTIVE = (
    "mv {h} /tmp/x",
    "mv {h}/ /tmp/x",
    "mv ~/.hermes ~/old",
    "rm -rf ~/.hermes/*",
    "rm {h}/state.db",
    "rm {h}/state.db-wal",
    'rm "$HERMES_HOME/state.db"',
    "rm -rf ~/.hermes/sessions",
    "unlink {h}/state.db",
    "shred -u {h}/state.db",
    "truncate -s0 {h}/state.db",
    "> {h}/state.db",
    "tee {h}/state.db </dev/null",
    "cp /dev/null {h}/state.db",
    "dd if=/dev/zero of={h}/state.db",
    'sqlite3 {h}/state.db "delete from messages where 1=1"',
    'sqlite3 {h}/state.db "update messages set content=0"',
    "rm -rf {h}/vault",
    "sudo rm {h}/state.db",
    "cd /tmp && rm {h}/state.db-journal",
    "ls; mv {h}/sessions /tmp/s",
    "echo x | tee -a {h}/state.db",
    'sqlite3 {h}/state.db "vacuum"',
)

_BENIGN = (
    "cp {h}/state.db /tmp/backup.db",
    'sqlite3 {h}/state.db "select count(*) from messages"',
    "sqlite3 {h}/state.db .schema",
    "cat {h}/state.db",
    "ls {h}",
    "cd {h} && git status",
    "rm {h}/cache/foo",
    "rm {h}/logs/agent.log",
    "mv {h}2 /tmp/x",
    "truncate -s0 {h}2/state.db",
    "rm ~/.hermes-backup/state.db",
    "mv {h}/skills/a {h}/skills/b",
    # A verb word that is data, not the command (searching transcripts, commit messages, prose).
    "grep -rl unlink {h}/sessions",
    'rg -n "mv " {h}/sessions/',
    "grep -c truncate {h}/sessions/s.jsonl",
    'git commit -m "{rm} {h}/state.db"',
    'echo "never {rm} {h}/state.db"',
    "man {rm} > {h}/sessions/x",
    # Neighbours of a store are not the store: the documented backup step and its cleanup.
    "cp {h}/state.db {h}/state.db.bak",
    "{rm} {h}/state.db.bak",
    "{rm} -f {h}/state.db.backup-20261101",
    "{rm} {h}/sessions-old.tar",
    "echo x | tee {h}/sessions-grep.txt",
    # Moving something INTO the home and the VACUUM INTO backup destroy nothing.
    "mv /tmp/skill {h}/",
    "mv /tmp/skill ~/.hermes",
    'sqlite3 {h}/state.db "VACUUM INTO \'/tmp/b.db\'"',
)


@pytest.mark.parametrize("template", _DESTRUCTIVE)
def test_destructive_verb_on_hermes_store_prompts(home, template):
    command = template.format(h=home, rm="r" + "m")
    dangerous, key, _ = detect_dangerous_command(command)
    assert dangerous is True, command
    assert key is not None


@pytest.mark.parametrize("template", _BENIGN)
def test_reads_and_unprotected_paths_stay_clean(home, template):
    command = template.format(h=home, rm="r" + "m")
    assert detect_dangerous_command(command) == (False, None, None), command


def test_repeated_verb_words_stay_linear(home):
    # The unanchored store rules rescanned from every repeated verb word (quadratic); anchored at a
    # command position they are tried only at command starts.
    command = "sqlite3 " * 8000 + home + "/state.db;x"
    started = time.monotonic()
    detect_dangerous_command(command)
    assert time.monotonic() - started < 6
