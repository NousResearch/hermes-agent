import yaml
from tools import approval


def test_legacy_string_allowlist_recovers_only_string_lists(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    description = "script execution via -e/-c flag"
    for value in ([description], yaml.safe_dump([description])):
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({"command_allowlist": value}))
        assert approval.load_permanent_allowlist() == {description}
        assert approval.is_approved("probe", description)
    assert "command_allowlist" in caplog.text


def test_malformed_allowlist_does_not_grant_approval(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for value in ("plain text", "[bad", {"not": "a list"}, [True, "candidate"], "[true, candidate]", 42):
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({"command_allowlist": value}))
        assert approval.load_permanent_allowlist() == set()
        assert not approval.is_approved("probe", "candidate")
    assert "command_allowlist" in caplog.text


def test_a_legacy_scalar_survives_a_save(tmp_path, monkeypatch):
    """The write path must parse `command_allowlist` the same way the read path does.

    Loading recovers the legacy scalar form but does not rewrite the file, so the
    scalar is still on disk when the next `always` calls save_permanent_allowlist().
    A save that does a bare set() on it iterates the string into characters and
    persists them over the operator's standing approvals -- including a bare '*',
    which the permanent-allowlist matcher then treats as a glob that approves
    every command.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"command_allowlist": yaml.safe_dump(["git status", "ls *"])}))

    approval.load_permanent_allowlist()
    approval.approve_permanent("docker *")
    approval.save_permanent_allowlist(approval._permanent_approved)

    assert sorted(yaml.safe_load(cfg.read_text())["command_allowlist"]) == [
        "docker *",
        "git status",
        "ls *",
    ]
    assert approval.is_approved("probe", "git status")


def test_a_malformed_allowlist_is_not_overwritten_by_a_save(tmp_path, monkeypatch):
    """A save must refuse a malformed on-disk value rather than iterate it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"command_allowlist": 42}))

    approval.load_permanent_allowlist()
    approval.approve_permanent("docker *")
    approval.save_permanent_allowlist(approval._permanent_approved)

    assert yaml.safe_load(cfg.read_text())["command_allowlist"] == 42
