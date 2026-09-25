"""Tests for file write safety and HERMES_WRITE_SAFE_ROOT sandboxing.

Based on PR #1085 by ismoilh (salvaged).
"""

import os
from pathlib import Path

import pytest

from agent.file_safety import is_write_denied as _is_write_denied


class TestStaticDenyList:
    """Basic sanity checks for the static write deny list."""

    def test_temp_file_not_denied_by_default(self, tmp_path: Path):
        target = tmp_path / "regular.txt"
        assert _is_write_denied(str(target)) is False


    def test_etc_shadow_is_denied(self):
        assert _is_write_denied("/etc/shadow") is True


class TestSshConfigApprovalGate:
    """~/.ssh/config is approval-gated, not hard-denied (private keys stay denied)."""

    def test_ssh_config_not_hard_denied(self):
        from agent.file_safety import is_write_denied

        # The client config carries no key material — it must NOT be in the
        # flat credential deny (it is routed through approval instead).
        assert is_write_denied(os.path.expanduser("~/.ssh/config")) is False

    def test_ssh_config_get_write_denied_error_is_none(self):
        from agent.file_safety import get_write_denied_error

        assert get_write_denied_error(os.path.expanduser("~/.ssh/config")) is None

    def test_ssh_config_is_approval_required(self):
        from agent.file_safety import is_write_approval_required

        assert is_write_approval_required(os.path.expanduser("~/.ssh/config")) is True

    def test_private_keys_still_hard_denied(self):
        from agent.file_safety import is_write_approval_required, is_write_denied

        for name in ("id_rsa", "id_ed25519", "authorized_keys"):
            p = os.path.expanduser(f"~/.ssh/{name}")
            assert is_write_denied(p) is True, name
            # A hard-denied credential is not merely approval-gated.
            assert is_write_approval_required(p) is False, name

    def test_other_ssh_dir_files_still_hard_denied(self):
        from agent.file_safety import is_write_denied

        # The ~/.ssh/ directory prefix deny still covers everything else,
        # e.g. a known_hosts or an arbitrary key file.
        assert is_write_denied(os.path.expanduser("~/.ssh/id_rsa.pub")) is True
        assert is_write_denied(os.path.expanduser("~/.ssh/secret_key")) is True

    def test_regular_file_not_approval_required(self, tmp_path: Path):
        from agent.file_safety import is_write_approval_required

        assert is_write_approval_required(str(tmp_path / "notes.txt")) is False



class TestSafeWriteRoot:
    """HERMES_WRITE_SAFE_ROOT should sandbox writes to a specific subtree."""

    def test_writes_inside_safe_root_are_allowed(self, tmp_path: Path, monkeypatch):
        safe_root = tmp_path / "workspace"
        child = safe_root / "subdir" / "file.txt"
        os.makedirs(child.parent, exist_ok=True)

        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe_root))
        assert _is_write_denied(str(child)) is False


    def test_safe_root_with_tilde_expansion(self, tmp_path: Path, monkeypatch):
        """~ in HERMES_WRITE_SAFE_ROOT should be expanded."""
        # Use a real subdirectory of tmp_path so we can test tilde-style paths
        safe_root = tmp_path / "workspace"
        inside = safe_root / "file.txt"
        os.makedirs(safe_root, exist_ok=True)

        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe_root))
        assert _is_write_denied(str(inside)) is False

    def test_safe_root_does_not_override_static_deny(self, tmp_path: Path, monkeypatch):
        """Even if a static-denied path is inside the safe root, it's still denied."""
        # Point safe root at home to include ~/.ssh
        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", os.path.expanduser("~"))
        assert _is_write_denied(os.path.expanduser("~/.ssh/id_rsa")) is True


class TestMultipleSafeWriteRoots:
    """HERMES_WRITE_SAFE_ROOT with multiple colon-separated directories."""

    def test_write_inside_first_root_allowed(self, tmp_path: Path, monkeypatch):
        root_a = tmp_path / "workspace_a"
        root_b = tmp_path / "workspace_b"
        child = root_a / "subdir" / "file.txt"
        os.makedirs(child.parent, exist_ok=True)
        os.makedirs(root_b, exist_ok=True)

        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", f"{root_a}{os.pathsep}{root_b}")
        assert _is_write_denied(str(child)) is False


    def test_trailing_separator_ignored(self, tmp_path: Path, monkeypatch):
        root = tmp_path / "workspace"
        inside = root / "file.txt"
        os.makedirs(root, exist_ok=True)

        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", f"{root}{os.pathsep}")
        assert _is_write_denied(str(inside)) is False


    def test_static_deny_still_wins_with_multiple_roots(self, tmp_path: Path, monkeypatch):
        """Static deny list takes priority even when multiple safe roots include home."""
        root = tmp_path / "workspace"
        os.makedirs(root, exist_ok=True)

        monkeypatch.setenv(
            "HERMES_WRITE_SAFE_ROOT",
            f"{root}{os.pathsep}{os.path.expanduser('~')}",
        )
        assert _is_write_denied(os.path.expanduser("~/.ssh/id_rsa")) is True

    def test_duplicate_roots_deduplicated(self, tmp_path: Path, monkeypatch):
        root = tmp_path / "workspace"
        inside = root / "file.txt"
        os.makedirs(root, exist_ok=True)

        monkeypatch.setenv(
            "HERMES_WRITE_SAFE_ROOT",
            f"{root}{os.pathsep}{root}",
        )
        assert _is_write_denied(str(inside)) is False


class TestGetWriteDeniedError:
    """get_write_denied_error() should distinguish credential vs safe-root blocks."""

    def test_credential_path_message(self):
        from agent.file_safety import get_write_denied_error

        err = get_write_denied_error(os.path.expanduser("~/.ssh/id_rsa"))
        assert err is not None
        assert "protected system/credential file" in err
        assert "HERMES_WRITE_SAFE_ROOT" not in err

    def test_safe_root_message(self, tmp_path: Path, monkeypatch):
        from agent.file_safety import get_write_denied_error

        safe_root = tmp_path / "workspace"
        outside = tmp_path / "outside.txt"
        os.makedirs(safe_root, exist_ok=True)

        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe_root))
        err = get_write_denied_error(str(outside))
        assert err is not None
        assert "outside HERMES_WRITE_SAFE_ROOT" in err
        assert str(safe_root) in err
        assert "protected system/credential file" not in err

    def test_allowed_path_returns_none(self, tmp_path: Path):
        from agent.file_safety import get_write_denied_error

        target = tmp_path / "ok.txt"
        assert get_write_denied_error(str(target)) is None


class TestSafeRootDenialMessageIntegration:
    """Regression tests verifying that file-tools surface the correct denial
    message when HERMES_WRITE_SAFE_ROOT blocks a path.

    Prior to this fix, ALL write denials returned the same "protected
    system/credential file" message regardless of root cause.  These tests
    exercise the actual write_file / patch_replace code path, not just
    the get_write_denied_error() helper in isolation.
    """

    @pytest.fixture
    def ops(self, tmp_path: Path):
        from tools.environments.local import LocalEnvironment
        from tools.file_operations import ShellFileOperations
        env = LocalEnvironment(cwd=str(tmp_path))
        return ShellFileOperations(env, cwd=str(tmp_path))

    def test_write_file_safe_root_outside_shows_safe_root_message(
        self, ops, tmp_path: Path, monkeypatch
    ):
        safe_root = tmp_path / "workspace"
        safe_root.mkdir()
        outside = tmp_path / "other" / "file.txt"
        outside.parent.mkdir()
        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe_root))

        res = ops.write_file(str(outside), "content")
        assert res.error is not None
        assert "outside HERMES_WRITE_SAFE_ROOT" in res.error
        assert str(safe_root) in res.error
        assert "credential" not in res.error
        assert not outside.exists()


    def test_write_file_credential_path_shows_credential_message(
        self, ops, tmp_path: Path
    ):
        res = ops.write_file("/etc/shadow", "content")
        assert res.error is not None
        assert "protected system/credential file" in res.error
        assert "outside" not in res.error

    def test_write_file_allowed_path_returns_no_error(
        self, ops, tmp_path: Path, monkeypatch
    ):
        safe_root = tmp_path / "workspace"
        safe_root.mkdir()
        inside = safe_root / "file.txt"
        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe_root))

        res = ops.write_file(str(inside), "content")
        assert res.error is None
        assert inside.read_text(encoding="utf-8") == "content"


class TestCheckSensitivePathMacOSBypass:
    """Verify _check_sensitive_path blocks /private/etc paths (issue #8734)."""

    @pytest.mark.platforms("linux")
    def test_etc_hosts_blocked(self):
        from tools.file_tools_write_guards import _check_sensitive_path
        assert _check_sensitive_path("/etc/hosts") is not None

    @pytest.mark.platforms("linux")
    def test_private_etc_hosts_blocked(self):
        from tools.file_tools_write_guards import _check_sensitive_path
        assert _check_sensitive_path("/private/etc/hosts") is not None

    @pytest.mark.platforms("linux")
    def test_private_etc_ssh_config_blocked(self):
        from tools.file_tools_write_guards import _check_sensitive_path
        assert _check_sensitive_path("/private/etc/ssh/sshd_config") is not None

    @pytest.mark.platforms("linux")
    def test_private_var_blocked(self):
        from tools.file_tools_write_guards import _check_sensitive_path
        assert _check_sensitive_path("/private/var/db/something") is not None

    @pytest.mark.platforms("linux")
    def test_boot_still_blocked(self):
        from tools.file_tools_write_guards import _check_sensitive_path
        assert _check_sensitive_path("/boot/grub/grub.cfg") is not None

    def test_safe_path_allowed(self):
        from tools.file_tools_write_guards import _check_sensitive_path
        assert _check_sensitive_path("/tmp/safe_file.txt") is None


class TestAtomicWrite:
    """write_file / patch land via a temp-file + atomic rename.

    The invariant: a write that fails partway NEVER corrupts the existing
    file, and the swap is a real rename (so a reader either sees the full
    old content or the full new content, never a half-written file). These
    run against a real LocalEnvironment so the actual shell script executes.
    """

    @pytest.fixture
    def ops(self, tmp_path: Path):
        from tools.environments.local import LocalEnvironment
        from tools.file_operations import ShellFileOperations
        env = LocalEnvironment(cwd=str(tmp_path))
        return ShellFileOperations(env, cwd=str(tmp_path))

    def test_overwrite_changes_inode(self, ops, tmp_path: Path):
        # A real rename allocates a new inode for the target; an in-place
        # rewrite would keep the same inode. This proves the swap is atomic.
        target = tmp_path / "f.txt"
        target.write_text("v1", encoding="utf-8")
        ino_before = os.stat(target).st_ino
        res = ops.write_file(str(target), "v2 content")
        assert res.error is None, res.error
        assert target.read_text(encoding="utf-8") == "v2 content"
        assert os.stat(target).st_ino != ino_before


    def test_no_temp_file_leaked_on_success(self, ops, tmp_path: Path):
        target = tmp_path / "f.txt"
        ops.write_file(str(target), "hello\n")
        assert [p for p in os.listdir(tmp_path) if ".hermes-tmp" in p] == []


    @pytest.mark.platforms("linux")
    def test_patch_routes_through_atomic_write(self, ops, tmp_path: Path):
        target = tmp_path / "edit.py"
        target.write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
        os.chmod(target, 0o600)
        res = ops.patch_replace(str(target), "b = 2", "b = 22")
        assert res.success, res.error
        assert target.read_text(encoding="utf-8") == "a = 1\nb = 22\nc = 3\n"
        assert (os.stat(target).st_mode & 0o777) == 0o600


class TestBomHandling:
    """UTF-8 BOM is stripped on read and preserved across write/patch.

    A BOM (U+FEFF, bytes EF BB BF) is an invisible leading marker some
    Windows editors prepend. The agent should never see it in read output,
    but a file that had one on disk must keep it after an edit so the byte
    signature is preserved.
    """

    BOM = "\ufeff"

    @pytest.fixture
    def ops(self, tmp_path: Path):
        from tools.environments.local import LocalEnvironment
        from tools.file_operations import ShellFileOperations
        env = LocalEnvironment(cwd=str(tmp_path))
        return ShellFileOperations(env, cwd=str(tmp_path))


    def test_read_strips_bom(self, ops, tmp_path: Path):
        target = tmp_path / "bom.py"
        # Write raw bytes with a real UTF-8 BOM prefix.
        target.write_bytes(self.BOM.encode("utf-8") + b"import os\nx = 1\n")
        res = ops.read_file(str(target))
        assert res.error is None, res.error
        # Line 1 content must NOT carry the phantom U+FEFF.
        first_line = res.content.split("\n", 1)[0]
        assert self.BOM not in first_line
        assert first_line.endswith("import os")


    def test_patch_matches_first_line_through_bom(self, ops, tmp_path: Path):
        # The whole point: an edit targeting the BOM-prefixed first line
        # must match cleanly (the matcher sees BOM-stripped content).
        target = tmp_path / "mod.py"
        target.write_bytes(self.BOM.encode("utf-8") + b"import os\nimport sys\n")
        res = ops.patch_replace(str(target), "import os", "import os, json")
        assert res.success, res.error
        raw = target.read_bytes()
        assert raw == self.BOM.encode("utf-8") + b"import os, json\nimport sys\n"

    def test_v4a_update_preserves_bom_real_ops(self, ops, tmp_path: Path):
        # V4A UPDATE path against REAL ShellFileOperations. This is the one
        # provider path whose pre_content is BOM-STRIPPED (read_file_raw
        # strips before _apply_update forwards it), so it regresses if
        # _file_has_bom ever trusts pre_content instead of probing disk.
        # Regression for teknium1's review on PR #55661.
        target = tmp_path / "bom_v4a.py"
        target.write_bytes(self.BOM.encode("utf-8") + b"print('hello')\n")
        patch = (
            "*** Begin Patch\n"
            f"*** Update File: {target}\n"
            "@@\n"
            "-print('hello')\n"
            "+print('world')\n"
            "*** End Patch"
        )
        res = ops.patch_v4a(patch)
        assert res.success, res.error
        raw = target.read_bytes()
        assert raw.startswith(self.BOM.encode("utf-8")), "BOM lost on V4A update"
        assert b"print('world')" in raw


    def test_v4a_update_keeps_terminal_escape_bytes_on_untouched_lines(self, ops, tmp_path: Path):
        # read_file_raw feeds the V4A write-back: every byte on a line the patch never touched
        # survives, including OSC escapes, BEL and literal fence-marker text.
        target = tmp_path / "prompt.sh"
        original = (b'set_title() { printf "\x1b]0;%s\x07" "$1"; }\n'
                    b'beep() { printf "\x07"; }\n'
                    b'SENTINEL = "__HERMES_FENCE_a9f7b3__\x07"  # marker text is file content too\n'
                    b'VERSION=1\n')
        target.write_bytes(original)
        patch = (
            "*** Begin Patch\n"
            f"*** Update File: {target}\n"
            "@@\n"
            "-VERSION=1\n"
            "+VERSION=2\n"
            "*** End Patch"
        )
        res = ops.patch_v4a(patch)
        assert res.success, res.error
        assert target.read_bytes() == original.replace(b"VERSION=1", b"VERSION=2")

    @pytest.mark.parametrize("mode", ["replace", "v4a"])
    def test_edit_keeps_bytes_utf8_cannot_decode_on_untouched_lines(self, ops, tmp_path: Path, mode):
        # Both edit paths write back every line they did not touch, so their source read must be
        # byte-exact: the text transport decodes with errors="replace", which turned this legacy
        # latin-1 byte into U+FFFD on disk. (Past the 1000-byte sample, where V4A reads it as text.)
        target = tmp_path / "legacy.py"
        original = (b"# -*- coding: latin-1 -*-\n" + b"# " + b"x" * 1100 + b"\n"
                    b"name = 'caf\xe9'\n"
                    b"x = 1\n")
        target.write_bytes(original)
        if mode == "replace":
            res = ops.patch_replace(str(target), "x = 1", "x = 2")
        else:
            res = ops.patch_v4a(f"*** Begin Patch\n*** Update File: {target}\n@@\n-x = 1\n+x = 2\n*** End Patch")
        assert res.success, res.error
        assert target.read_bytes() == original.replace(b"x = 1", b"x = 2")
        # The file declares its encoding, so it is valid Python and lints clean.
        lints = res.lint.values() if mode == "v4a" else [res.lint]
        assert [lint["status"] for lint in lints] == ["ok"], res.lint

    @staticmethod
    def _noisy_env():
        """A real shell whose merged stdout carries a connect banner, as remote backends do."""
        from tools.environments.local import LocalEnvironment

        class Env(LocalEnvironment):
            def execute(self, command, *args, **kwargs):
                result = super().execute(command, *args, **kwargs)
                if isinstance(result, dict):
                    result = dict(result)
                    result["output"] = "TERM\n" + (result.get("output") or "")
                return result
        return Env

    def test_byte_exact_read_fences_backend_stdout_noise(self, tmp_path: Path, monkeypatch):
        # The edit paths write this read straight back, so noise in the backend's merged stdout must
        # never reach the decode. "TERM" is four base64 characters: unfenced it decodes to b"LDL" and
        # lands at the head of the file. Remote backends announce things on connect, so the local
        # native fast path is off here and the base64 transport is what runs.
        from tools.file_operations import ShellFileOperations
        monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "0")

        target = tmp_path / "conf.txt"
        original = b"HEADER\nVERSION=1\n"
        target.write_bytes(original)
        ops = ShellFileOperations(self._noisy_env()(cwd=str(tmp_path)), cwd=str(tmp_path))

        assert ops._read_exact_bytes(str(target)) == (original, None)
        ops.patch_replace(str(target), "VERSION=1", "VERSION=2")
        assert target.read_bytes() == original.replace(b"VERSION=1", b"VERSION=2")

    def test_binary_admission_sample_fences_backend_stdout_noise(self, tmp_path: Path, monkeypatch):
        # The same transport gap one step EARLIER: _sample_file_bytes is the binary-admission gate
        # in front of the byte-exact read, so backend noise joined onto its base64 decides whether a
        # file is editable at all and what a refusal reports about it. The sample must be the file's
        # own leading bytes, not the backend's banner decoded into them.
        from tools.file_operations import ShellFileOperations
        monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "0")

        target = tmp_path / "head.bin"
        original = b"\x00\x01\x02binary payload\n"
        target.write_bytes(original)
        ops = ShellFileOperations(self._noisy_env()(cwd=str(tmp_path)), cwd=str(tmp_path))

        assert ops._sample_file_bytes(str(target)) == original

    @staticmethod
    def _env_without(*missing: str):
        """A real shell where only the named BINARIES are absent (busybox, distroless)."""
        import re as _re
        from tools.environments.local import LocalEnvironment
        stub = "( echo 'sh: not found' >&2; exit 127 )"  # a SUBSHELL: `exit` must not kill the shell

        class Env(LocalEnvironment):
            def execute(self, command, *args, **kwargs):
                for name in missing:
                    command = _re.sub(rf"\b{name} <", f"{stub} <", command)
                    command = _re.sub(rf"\b{name}\b(?! <)", stub, command)
                return super().execute(command, *args, **kwargs)
        return Env

    def test_byte_exact_read_falls_back_to_hex_without_base64(self, tmp_path: Path, monkeypatch):
        # base64 is not on every backend. The sample path already degrades when it is missing
        # (_detect_binary), so the byte-exact read must too, and byte-exactly.
        from tools.file_operations import ShellFileOperations
        monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "0")
        target = tmp_path / "conf.txt"
        original = b"HEADER\nVERSION=1\n"
        target.write_bytes(original)
        ops = ShellFileOperations(self._env_without("base64")(cwd=str(tmp_path)), cwd=str(tmp_path))

        assert ops._read_exact_bytes(str(target)) == (original, None)
        assert ops.patch_replace(str(target), "VERSION=1", "VERSION=2").success
        assert target.read_bytes() == original.replace(b"VERSION=1", b"VERSION=2")

    def test_add_file_refuses_when_the_read_failed_rather_than_the_path_being_free(
            self, tmp_path: Path, monkeypatch):
        # `Add File` uses read_file_raw's error as its existence check. A backend with no byte
        # transport at all makes that read FAIL, which must not read as "the path is free" —
        # that writes the Add payload over the file the check exists to protect.
        from tools.file_operations import ShellFileOperations
        monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "0")
        target = tmp_path / "KEEP.txt"
        precious = b"KEEP ME\n"
        target.write_bytes(precious)
        ops = ShellFileOperations(self._env_without("base64", "od")(cwd=str(tmp_path)), cwd=str(tmp_path))

        read = ops.read_file_raw(str(target))
        assert read.error and not read.not_found  # a failed read, NOT an absent path
        res = ops.patch_v4a(f"*** Begin Patch\n*** Add File: {target}\n+clobbered\n*** End Patch")
        assert not res.success
        assert target.read_bytes() == precious

    def test_move_refuses_a_destination_it_could_not_read_before_any_op_applies(
            self, tmp_path: Path, monkeypatch):
        # Validation must keep "the read failed" apart from "the path is absent" for a Move
        # destination too, and the apply must re-check it before `mv` replaces whatever is there.
        from tools.file_operations import ShellFileOperations
        monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "0")
        dst = tmp_path / "dst.txt"
        dst.write_bytes(b"PRECIOUS DESTINATION\n")
        ops = ShellFileOperations(self._env_without("base64", "od")(cwd=str(tmp_path)), cwd=str(tmp_path))

        res = ops.patch_v4a(
            "*** Begin Patch\n"
            f"*** Add File: {tmp_path / 'new-src.txt'}\n+SOURCE\n"
            f"*** Move File: {tmp_path / 'new-src.txt'} -> {dst}\n"
            "*** End Patch")
        assert not res.success
        assert dst.read_bytes() == b"PRECIOUS DESTINATION\n"
        assert not (tmp_path / "new-src.txt").exists()

    @pytest.mark.parametrize("transport,noise", [("base64", "TERM"), ("od", "4c 44")])
    def test_output_inside_the_byte_exact_read_never_reaches_a_write(
            self, tmp_path: Path, monkeypatch, transport, noise):
        # The fence drops noise around the read, not noise printed WHILE it runs: a BASH_ENV DEBUG
        # hook firing for the transport command alone puts text inside the payload that still
        # decodes ("TERM" is b"LDL"; "4c 44" is hex). Such a read must fail, and no edit may write.
        from tools.file_operations import ShellFileOperations
        monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "0")
        hook = tmp_path / "hook.sh"
        hook.write_text(f"trap '[[ $BASH_COMMAND == {transport}* ]] && echo \"{noise}\"' DEBUG\n")
        target = tmp_path / "conf.txt"
        original = b"HEADER\nVERSION=1\n"
        target.write_bytes(original)
        missing = ("base64",) if transport == "od" else ()
        env = self._env_without(*missing)(cwd=str(tmp_path), env={"BASH_ENV": str(hook)})
        ops = ShellFileOperations(env, cwd=str(tmp_path))

        assert ops._read_exact_bytes(str(target))[0] is None
        assert not ops.patch_replace(str(target), "VERSION=1", "VERSION=2").success
        assert not ops.patch_v4a(
            f"*** Begin Patch\n*** Update File: {target}\n@@\n-VERSION=1\n+VERSION=2\n*** End Patch").success
        assert target.read_bytes() == original

    @pytest.mark.parametrize("op", ["add", "move"])
    def test_a_dangling_symlink_destination_is_occupied(self, ops, tmp_path: Path, op):
        # `[ -f ]` and `[ -e ]` follow the link, so a dangling one read as an absent path: Add
        # followed it and created its target, Move replaced the link. The entry is there.
        link = tmp_path / "link.txt"
        link.symlink_to(tmp_path / "gone.txt")
        (tmp_path / "src.txt").write_bytes(b"SOURCE\n")
        body = (f"*** Add File: {link}\n+X\n" if op == "add"
                else f"*** Move File: {tmp_path / 'src.txt'} -> {link}\n")
        res = ops.patch_v4a(f"*** Begin Patch\n{body}*** End Patch")
        assert not res.success
        assert link.is_symlink() and os.readlink(link) == str(tmp_path / "gone.txt")
        assert not (tmp_path / "gone.txt").exists()
        assert (tmp_path / "src.txt").read_bytes() == b"SOURCE\n"

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX only: needs os.mkfifo and SIGALRM")
    def test_native_byte_exact_read_never_opens_a_non_regular_file(self, tmp_path: Path, monkeypatch):
        # The native fast path bypasses the backend timeout, so a blocking open there hangs the
        # thread with nothing to interrupt it. The shell path below has a timeout and is allowed
        # to take a FIFO; the native path must hand it over instead of opening it. Stubbing the
        # shell read keeps this about the native branch: if it opens the FIFO the test hangs.
        import signal
        from tools.file_operations import ExecuteResult, ShellFileOperations
        from tools.environments.local import LocalEnvironment
        fifo = tmp_path / "pipe"
        os.mkfifo(fifo)  # no writer: a blocking open never returns
        ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)), cwd=str(tmp_path))
        monkeypatch.setattr(ops, "_exec",
                            lambda *a, **k: ExecuteResult(stdout="handed to the shell", exit_code=1))

        def _bail(*_args):
            raise TimeoutError("the native read opened a FIFO and blocked")
        previous = signal.signal(signal.SIGALRM, _bail)
        signal.alarm(5)
        try:
            data, failed = ops._read_exact_bytes(str(fifo))
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
        assert data is None and failed is not None and "handed to the shell" in failed.stdout


class TestProtectedInstructionFiles:
    """Writes to agent-instruction files ALWAYS require approval.

    AGENTS.md / CLAUDE.md / SOUL.md / .cursorrules / project-local .hermes
    config steer future agent behavior, so a prompt-injected agent writing
    them is a persistence vector. The gate must ask the human every time —
    even under yolo/auto-approve — and fail closed when no human channel
    exists. Ported from: RooCodeInc/Roo-Code RooProtectedController
    (Apache-2.0); symlink lesson from #41351.
    """

    @pytest.fixture(autouse=True)
    def _gate_on(self, monkeypatch):
        import tools.file_tools_write_guards as ft
        monkeypatch.setattr(
            ft, "_protected_instruction_config", lambda: (True, [])
        )
        yield

    @pytest.fixture
    def approvals(self, monkeypatch):
        """Install a CLI approval callback; record calls; scripted answers."""
        from tools.terminal_tool import set_approval_callback
        state = {"calls": [], "answer": "deny"}

        def cb(command, description, **kwargs):
            state["calls"].append(
                {"command": command, "description": description, **kwargs}
            )
            return state["answer"]

        set_approval_callback(cb)
        yield state
        set_approval_callback(None)

    def _write(self, path, content="injected"):
        import json
        from tools.file_tools import write_file_tool
        return json.loads(write_file_tool(str(path), content))

    # ---- core behavior -------------------------------------------------

    @pytest.mark.parametrize(
        "name", ["AGENTS.md", "CLAUDE.md", "SOUL.md", ".cursorrules"]
    )
    def test_deny_blocks_write(self, tmp_path, approvals, name):
        target = tmp_path / name
        approvals["answer"] = "deny"
        res = self._write(target)
        assert res.get("error"), res
        assert "BLOCKED" in res["error"]
        assert not target.exists()
        assert len(approvals["calls"]) == 1

    def test_approve_once_allows_write(self, tmp_path, approvals):
        target = tmp_path / "AGENTS.md"
        approvals["answer"] = "once"
        res = self._write(target, "approved content")
        assert not res.get("error"), res
        assert target.read_text(encoding="utf-8") == "approved content"
        assert len(approvals["calls"]) == 1

    def test_prompts_even_under_yolo(self, tmp_path, approvals, monkeypatch):
        """The whole point: auto-approve/yolo must NOT bypass this gate."""
        import tools.approval as A
        monkeypatch.setattr(A, "_YOLO_MODE_FROZEN", True)
        target = tmp_path / "AGENTS.md"
        approvals["answer"] = "deny"
        res = self._write(target)
        assert res.get("error") and "BLOCKED" in res["error"]
        assert not target.exists()
        assert len(approvals["calls"]) == 1, "yolo bypassed the protected gate"

    def test_second_write_prompts_again(self, tmp_path, approvals):
        """One-operation approval: no session stickiness."""
        target = tmp_path / "AGENTS.md"
        approvals["answer"] = "once"
        self._write(target)
        self._write(target, "second")
        assert len(approvals["calls"]) == 2

    def test_cli_prompt_is_told_no_scope_persists(self, tmp_path, approvals):
        """The prompt must not advertise a scope this gate discards.

        Since nothing is persisted, a rendered "session"/"always" option
        re-prompts on the very next write and reads as a broken gate
        (#81887).
        """
        approvals["answer"] = "once"
        self._write(tmp_path / "SOUL.md")
        call = approvals["calls"][0]
        assert call["allow_session"] is False
        assert call["allow_permanent"] is False

    def test_regular_file_never_prompts(self, tmp_path, approvals):
        res = self._write(tmp_path / "notes.md", "hello")
        assert not res.get("error"), res
        assert approvals["calls"] == []

    def test_no_human_fails_closed(self, tmp_path):
        # No approval callback registered, not gateway → block, don't hang.
        target = tmp_path / "AGENTS.md"
        res = self._write(target)
        assert res.get("error") and "BLOCKED" in res["error"]
        assert not target.exists()

    def test_config_disabled_skips_gate(self, tmp_path, approvals, monkeypatch):
        import tools.file_tools_write_guards as ft
        monkeypatch.setattr(
            ft, "_protected_instruction_config", lambda: (False, [])
        )
        res = self._write(tmp_path / "AGENTS.md", "ok")
        assert not res.get("error"), res
        assert approvals["calls"] == []

    def test_extra_patterns_from_config(self, tmp_path, approvals, monkeypatch):
        import tools.file_tools_write_guards as ft
        monkeypatch.setattr(
            ft, "_protected_instruction_config", lambda: (True, ["*.mdc"])
        )
        approvals["answer"] = "deny"
        res = self._write(tmp_path / "rules.mdc")
        assert res.get("error") and "BLOCKED" in res["error"]

    # ---- adversarial path shapes ----------------------------------------

    @pytest.mark.require_symlinks
    def test_symlink_to_protected_file_is_gated(self, tmp_path, approvals):
        """#41351 lesson: realpath first — innocent name, protected target."""
        real = tmp_path / "AGENTS.md"
        real.write_text("original", encoding="utf-8")
        link = tmp_path / "innocent.txt"
        link.symlink_to(real)
        approvals["answer"] = "deny"
        res = self._write(link, "injected")
        assert res.get("error") and "BLOCKED" in res["error"]
        assert real.read_text(encoding="utf-8") == "original"

    def test_case_variant_is_gated(self, tmp_path, approvals):
        approvals["answer"] = "deny"
        res = self._write(tmp_path / "agents.MD")
        assert res.get("error") and "BLOCKED" in res["error"]

    def test_relative_traversal_is_gated(self, tmp_path, approvals, monkeypatch):
        (tmp_path / "x").mkdir()
        monkeypatch.chdir(tmp_path)
        approvals["answer"] = "deny"
        res = self._write("./x/../AGENTS.md")
        assert res.get("error") and "BLOCKED" in res["error"]
        assert not (tmp_path / "AGENTS.md").exists()

    def test_arbitrary_directory_basename_is_gated(self, tmp_path, approvals):
        """Any-directory scope: project-context files load from cwd trees."""
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        approvals["answer"] = "deny"
        res = self._write(deep / "CLAUDE.md")
        assert res.get("error") and "BLOCKED" in res["error"]

    def test_project_local_hermes_dir_is_gated(self, tmp_path, approvals):
        proj = tmp_path / "proj" / ".hermes"
        proj.mkdir(parents=True)
        approvals["answer"] = "deny"
        res = self._write(proj / "config.yaml")
        assert res.get("error") and "BLOCKED" in res["error"]

    def test_checkout_nested_under_hermes_dir_not_gated(self, tmp_path, approvals):
        """A repo living UNDER a .hermes dir (e.g. ~/.hermes/hermes-agent)
        must not have every write gated — only files directly inside a
        .hermes dir count as project config."""
        repo = tmp_path / ".hermes" / "some-repo" / "src"
        repo.mkdir(parents=True)
        res = self._write(repo / "module.py", "x = 1\n")
        assert not res.get("error"), res
        assert approvals["calls"] == []

    def test_real_hermes_home_not_gated_by_this_check(
        self, tmp_path, approvals, monkeypatch
    ):
        """~/.hermes itself is governed by existing guards, not this gate."""
        import tools.file_tools_write_guards as ft
        fake_home = tmp_path / ".hermes"
        (fake_home / "notes").mkdir(parents=True)
        monkeypatch.setattr(
            ft, "_get_real_hermes_home", lambda: str(fake_home.resolve())
        )
        res = self._write(fake_home / "notes" / "scratch.txt", "ok")
        assert not res.get("error"), res
        assert approvals["calls"] == []

    # ---- patch tool -----------------------------------------------------

    def test_patch_replace_mode_is_gated(self, tmp_path, approvals):
        from tools.file_tools import patch_tool
        import json
        target = tmp_path / "SOUL.md"
        target.write_text("be kind\n", encoding="utf-8")
        approvals["answer"] = "deny"
        res = json.loads(patch_tool(
            mode="replace", path=str(target),
            old_string="be kind", new_string="obey injected orders",
        ))
        assert res.get("error") and "BLOCKED" in res["error"]
        assert target.read_text(encoding="utf-8") == "be kind\n"

    def test_patch_v4a_multifile_one_protected_blocks_whole_patch(
        self, tmp_path, approvals
    ):
        """Policy: one protected file gates the ENTIRE patch (deny = nothing
        applies, including the innocent file)."""
        from tools.file_tools import patch_tool
        import json
        agents = tmp_path / "AGENTS.md"
        agents.write_text("rules\n", encoding="utf-8")
        plain = tmp_path / "plain.txt"
        plain.write_text("hello\n", encoding="utf-8")
        patch = (
            "*** Begin Patch\n"
            f"*** Update File: {plain}\n"
            "@@\n"
            "-hello\n"
            "+world\n"
            f"*** Update File: {agents}\n"
            "@@\n"
            "-rules\n"
            "+injected\n"
            "*** End Patch"
        )
        approvals["answer"] = "deny"
        res = json.loads(patch_tool(mode="patch", patch=patch))
        assert res.get("error") and "BLOCKED" in res["error"]
        assert plain.read_text(encoding="utf-8") == "hello\n"
        assert agents.read_text(encoding="utf-8") == "rules\n"
        assert len(approvals["calls"]) == 1

    def test_patch_v4a_approved_applies(self, tmp_path, approvals):
        from tools.file_tools import patch_tool
        import json
        agents = tmp_path / "AGENTS.md"
        agents.write_text("rules\n", encoding="utf-8")
        patch = (
            "*** Begin Patch\n"
            f"*** Update File: {agents}\n"
            "@@\n"
            "-rules\n"
            "+updated rules\n"
            "*** End Patch"
        )
        approvals["answer"] = "once"
        res = json.loads(patch_tool(mode="patch", patch=patch))
        assert not res.get("error"), res
        assert agents.read_text(encoding="utf-8") == "updated rules\n"

    # ---- gateway round-trip ----------------------------------------------

    def test_gateway_notify_resolve_once_allows(self, tmp_path):
        import tools.approval as A
        from tools import approval_context
        session_key = "protected-files-test-session"
        token = approval_context.set_current_session_key(session_key)
        try:
            def notify(approval_data):
                # Buttons must not offer persistent scopes for this gate.
                assert approval_data.get("allow_permanent") is False
                assert approval_data.get("allow_session") is False
                A.resolve_gateway_approval(session_key, "once")

            A.register_gateway_notify(session_key, notify)
            try:
                res = self._write(tmp_path / "AGENTS.md", "gateway approved")
                assert not res.get("error"), res
                assert (tmp_path / "AGENTS.md").read_text(encoding="utf-8") == "gateway approved"
            finally:
                A.unregister_gateway_notify(session_key)
        finally:
            approval_context.reset_current_session_key(token)

    def test_gateway_payload_renders_only_once_and_deny(self, tmp_path):
        """End-to-end: what this gate emits, a TUI/desktop client can render.

        The transport used to derive its button set from ``allow_permanent``
        alone, so it re-added a "session" scope the gate refuses to persist —
        users tapped it and got re-prompted on every write (#81887). Asserting
        the two layers together is what catches that drift.
        """
        import tools.approval as A
        from tools import approval_context
        from tui_gateway.server import _approval_request_payload

        session_key = "protected-files-payload-session"
        token = approval_context.set_current_session_key(session_key)
        rendered = {}
        try:
            def notify(approval_data):
                rendered.update(_approval_request_payload(approval_data))
                A.resolve_gateway_approval(session_key, "once")

            A.register_gateway_notify(session_key, notify)
            try:
                self._write(tmp_path / "SOUL.md", "gateway approved")
            finally:
                A.unregister_gateway_notify(session_key)
        finally:
            approval_context.reset_current_session_key(token)

        assert rendered["choices"] == ["once", "deny"]


class TestProfileHomeExemptsHermesRoot:
    """issue #60: under ``hermes -p <name>`` (``HERMES_HOME=<root>/profiles/<name>``)
    the exemption used to cover ONLY the profile dir, so the ROOT's direct files
    (LEDGER.md / MEMORY.md / SOUL.md ...) fell through to the ``.hermes`` component
    rule, were read as project-local ``.hermes`` config, and — having no approval
    channel headless — failed closed. That blocked #54 (LEDGER.md edit). The gate
    must exempt the whole Hermes tree, exactly like the default profile does.
    """

    @pytest.fixture(autouse=True)
    def _gate_on(self, monkeypatch):
        import tools.file_tools_write_guards as ft
        monkeypatch.setattr(
            ft, "_protected_instruction_config", lambda: (True, [])
        )
        # The resolved-home slot is filled once per process; keep the fixture honest.
        monkeypatch.setattr(ft, "_real_hermes_home_loaded", False)
        monkeypatch.setattr(ft, "_real_hermes_home_cached", None)
        yield

    @pytest.fixture
    def approvals(self, monkeypatch):
        from tools.terminal_tool import set_approval_callback
        state = {"calls": [], "answer": "deny"}

        def cb(command, description, **kwargs):
            state["calls"].append({"command": command, "description": description})
            return state["answer"]

        set_approval_callback(cb)
        yield state
        set_approval_callback(None)

    def _write(self, path, content="injected"):
        import json
        from tools.file_tools import write_file_tool
        return json.loads(write_file_tool(str(path), content))

    def _profile_layout(self, tmp_path: Path):
        """A real-shaped Hermes root: ``<tmp>/home/profiles/worker`` + root markers."""
        root = tmp_path / "home"
        profile = root / "profiles" / "worker"
        (profile / "workspace").mkdir(parents=True)
        (root / "config.yaml").write_text("model:\n  default: x\n", encoding="utf-8")
        return root, profile

    def test_named_profile_scope_exempts_root_direct_files(self, tmp_path, monkeypatch, approvals):
        """Under a named profile bound by the per-turn scope (multiplex path), the ROOT's own store is
        not project-local ``.hermes`` config: the write lands with no approval prompt."""
        import tools.file_tools_write_guards as ft
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override

        root, profile = self._profile_layout(tmp_path)
        monkeypatch.delenv("HERMES_HOME", raising=False)
        token = set_hermes_home_override(str(profile))
        try:
            assert os.path.realpath(str(root)) in ft._hermes_exempt_homes()
            for name in ("LEDGER.md", "MEMORY.md", "SOUL.md", "AGENTS.md"):
                assert ft._protected_instruction_reason(str(root / name)) is None, name
            res = self._write(root / "LEDGER.md", "caliber fixed")
        finally:
            reset_hermes_home_override(token)
        assert not res.get("error"), res
        assert (root / "LEDGER.md").read_text(encoding="utf-8") == "caliber fixed"
        assert approvals["calls"] == []

    def test_only_a_real_hermes_root_is_exempt(self, tmp_path, monkeypatch, approvals):
        """Negatives hold with a named profile active: a checkout's ``.hermes/config.yaml`` and
        protected basenames stay gated (fail-closed, unwritten), and a coincidental
        ``.../profiles/<name>`` tree that is NOT a Hermes root never exempts its parent."""
        import tools.file_tools_write_guards as ft
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override

        root, profile = self._profile_layout(tmp_path)
        repo = tmp_path / "repo"
        (repo / ".hermes").mkdir(parents=True)
        monkeypatch.delenv("HERMES_HOME", raising=False)
        token = set_hermes_home_override(str(profile))
        try:
            assert ft._protected_instruction_reason(str(repo / ".hermes" / "config.yaml"))
            assert ft._protected_instruction_reason(str(repo / "AGENTS.md")) == "AGENTS.md"
            target = repo / ".hermes" / "config.yaml"
            res = self._write(target, "gate: off\n")
        finally:
            reset_hermes_home_override(token)
        assert res.get("error") and "BLOCKED" in res["error"]
        assert not target.exists()
        assert len(approvals["calls"]) == 1

        fake_profile = tmp_path / "not-a-hermes-root" / "profiles" / "worker"
        fake_profile.mkdir(parents=True)
        token = set_hermes_home_override(str(fake_profile))
        try:
            assert ft._hermes_exempt_homes() == (os.path.realpath(str(fake_profile)),)
        finally:
            reset_hermes_home_override(token)


class TestMultiplexProfileWriteGuardsAreProfileScoped:
    """#107327: a multiplexed gateway scopes ``HERMES_HOME`` per turn via a
    contextvar. The home/config path getters must resolve per call, or whichever
    profile ran first in the process freezes both the protected-instruction gate
    and the ``config.yaml`` hard-block for every later profile — up to letting a
    later profile rewrite its own ``config.yaml`` the block exists to protect."""

    def _profiles(self, tmp_path: Path):
        root = tmp_path / "home"
        a, b = root / "profiles" / "alpha", root / "profiles" / "beta"
        for home in (a, b):
            (home / "workspace").mkdir(parents=True)
            (home / "config.yaml").write_text(
                "model:\n  default: original\n", encoding="utf-8"
            )
        return a, b

    def test_home_getter_tracks_active_profile_after_a_prior_scope(self, tmp_path):
        import tools.file_tools_write_guards as ft
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        a, b = self._profiles(tmp_path)
        # A normal alpha turn resolves (and, on the buggy path, would freeze) home.
        tok = set_hermes_home_override(str(a))
        try:
            assert ft._get_real_hermes_home() == os.path.realpath(str(a))
        finally:
            reset_hermes_home_override(tok)
        # The next turn is beta — the getter must now return beta's home, not alpha's.
        tok = set_hermes_home_override(str(b))
        try:
            assert ft._get_real_hermes_home() == os.path.realpath(str(b))
        finally:
            reset_hermes_home_override(tok)

    def test_config_hard_block_refuses_beta_config_even_after_alpha_turn(self, tmp_path):
        """End-to-end: the ``config.yaml`` hard-block must fire for beta's own
        config under beta's scope, regardless of alpha having run first."""
        import tools.file_tools_write_guards as ft
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        a, b = self._profiles(tmp_path)
        tok = set_hermes_home_override(str(a))
        try:
            ft._get_hermes_config_resolved()  # warm the (formerly poisoning) alpha lookup
        finally:
            reset_hermes_home_override(tok)

        tok = set_hermes_home_override(str(b))
        try:
            err = ft._check_sensitive_path(str(b / "config.yaml"), "default")
        finally:
            reset_hermes_home_override(tok)
        assert err is not None
        assert "Refusing to write to Hermes config file" in err
