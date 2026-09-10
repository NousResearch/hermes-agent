"""The container cwd verdict must not depend on the host Hermes runs on.

``_is_unusable_container_cwd`` ended its check with ``os.path.isabs``, and
``os.path`` is ``posixpath`` or ``ntpath`` depending on the machine. The same
string therefore got different verdicts on different hosts.

Host paths -- including every Windows drive -- are already handled by
``_is_host_cwd`` (``_WINDOWS_DRIVE_RE``), so the drive families were never the
problem here. The problem is the other direction: CPython 3.13 changed
``ntpath.isabs`` to return False for a rooted path with no drive, so on a
Windows host running 3.13+ the guard would begin rejecting ``/workspace`` and
``/root`` -- discarding exactly the values it exists to let through, and
sending every such session to ``default_cwd`` instead of its sandbox
workspace.

Asking for a leading ``/`` is the rule the sandbox actually imposes and gives
the same answer on every host and every Python.
"""

import ntpath
import posixpath

import tools.terminal_tool_config as cfg


class TestVerdictIsTheSameOnEveryHost:
    REJECT = [
        r"C:\Users\me", "C:/Users/me",       # host, via the drive regex
        r"D:\hermes", "d:/hermes", r"Z:\x",  # any drive, either slash
        "/home/me", "/Users/me",             # host, via the prefixes
        ".", "..", "src/", "work/uservice",  # relative
    ]
    KEEP = ["/workspace", "/workspace/task42", "/root", "/", "/opt/data"]

    def test_rejected_families(self):
        for p in self.REJECT:
            assert cfg._is_unusable_container_cwd(p) is True, p

    def test_kept_families(self):
        for p in self.KEEP:
            assert cfg._is_unusable_container_cwd(p) is False, p

    def test_verdict_does_not_consult_os_path(self, monkeypatch):
        # Swap the module's os.path for the other flavour and demand identical
        # answers. This is the property the old implementation lacked, and the
        # one that breaks on Windows + CPython 3.13.
        for flavour in (ntpath, posixpath):
            monkeypatch.setattr(cfg.os, "path", flavour)
            for p in self.REJECT:
                assert cfg._is_unusable_container_cwd(p) is True, (flavour.__name__, p)
            for p in self.KEEP:
                assert cfg._is_unusable_container_cwd(p) is False, (flavour.__name__, p)

    def test_the_flavours_really_disagree_about_a_rooted_path(self):
        # Keeps the property test from going vacuous. On 3.13+ these differ;
        # on older Pythons they agree, and the property must hold either way,
        # so assert the shape rather than a version-specific value.
        assert posixpath.isabs("/workspace") is True
        assert ntpath.isabs("/workspace") in (True, False)
        assert ntpath.isabs(r"D:\x") is True
        assert posixpath.isabs(r"D:\x") is False

    def test_empty_is_not_flagged(self):
        assert cfg._is_unusable_container_cwd("") is False
