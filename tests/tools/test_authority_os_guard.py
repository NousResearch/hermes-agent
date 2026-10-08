"""The write boundary is the process that opens the final path."""

import json
import os
import shlex
import stat
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path

import pytest

from tools.environments.base_session_env import _wrap_command_script
from tools.write_boundary import (
    clear_official_writers,
    clear_protected_basenames,
    guard_command,
    official_exec_argv,
    refuse_command,
    register_official_writer,
    register_protected_basenames,
)


NAMES = (
    "facts.json", "run-card.json", "trade_plan.json",
    "bundle.json", "validation.json", "publish_result.json",
)
OFFICIAL = (
    "python3 /root/.hermes/skills/finance/trading-agents/scripts/authority_write.py "
    "--kind facts --run-dir /tmp/run --relative facts.json --input /tmp/cand.json"
)
SPLICE = "name=facts; ext=json; printf x > ${name}.${ext}"
CONTAINER = os.environ.get("HERMES_TEST_CONTAINER", "hermes-ab10a2e0")


def _home(tmp_path):
    home = tmp_path / "big"
    home.mkdir()
    clear_protected_basenames()
    clear_official_writers()
    register_protected_basenames(str(home), NAMES)
    return home


def _bash(command: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["/bin/bash", "-c", command], cwd=cwd, capture_output=True, text=True,
    )


def _session(command: str, cwd: Path) -> str:
    """Same eval quoting the terminal environment uses around a user command."""
    return _wrap_command_script(
        command,
        quoted_cwd=shlex.quote(str(cwd)),
        quoted_snap=shlex.quote(str(cwd / "snap")),
        snap_tmp_template="/tmp/hermes-unused.XXXXXX",
        passthrough_names=(),
        snapshot_ready=False,
        cwd_marker="@@HERMES_CWD@@",
    )


@pytest.fixture(autouse=True)
def _clear_boundary():
    clear_protected_basenames()
    clear_official_writers()
    yield
    clear_protected_basenames()
    clear_official_writers()


def test_filter_refuses_syscalls_that_hide_the_path():
    """io_uring and open_by_handle_at have no path argument, so the filter denies them."""
    import errno

    from tools.authority_os_guard import AARCH64, FILTERS, SECCOMP_RET_ERRNO

    want = SECCOMP_RET_ERRNO | errno.EPERM
    words = [item.k for item in FILTERS]
    for name in (
        "open_by_handle_at", "io_uring_setup", "io_uring_enter", "io_uring_register",
        "mount", "unshare", "open_tree",
    ):
        number = AARCH64[name]
        assert any(
            words[index] == number and words[index + 1] == want
            for index in range(len(words) - 1)
        ), name


def test_exact_writer_argv_is_narrow(tmp_path):
    home = _home(tmp_path)
    register_official_writer("/root/.hermes/skills/finance/trading-agents/scripts/authority_write.py")
    assert official_exec_argv(OFFICIAL) is not None
    assert refuse_command(OFFICIAL, home=str(home)) is None
    assert official_exec_argv(OFFICIAL + "; printf x > /tmp/x") is None
    assert refuse_command(
        "python /root/.hermes/skills/finance/trading-agents/scripts/authority_write.py --kind facts",
        home=str(home),
    ) is None
    assert "facts.json" not in SPLICE
    assert refuse_command(SPLICE, home=str(home)) is None
    copied = OFFICIAL.replace("/root/.hermes/skills", "/tmp/skills")
    assert official_exec_argv(copied) is None


def test_an_unguarded_environment_refuses_instead_of_running(tmp_path):
    home = _home(tmp_path)
    assert guard_command("echo hi", env_type="ssh", home=str(home)) is None
    assert guard_command("echo hi", env_type="modal", home=str(home)) is None
    clear_protected_basenames()
    assert guard_command("echo hi", env_type="ssh", home=str(home)) == "echo hi"


def test_darwin_blocks_dynamic_writes_and_keeps_ordinary_files(tmp_path):
    if sys.platform != "darwin":
        pytest.skip("seatbelt runs on the host")
    home = _home(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    facts = work / "facts.json"
    facts.write_text("old-facts\n", encoding="utf-8")
    nested = work / "sub"
    nested.mkdir()
    (nested / "facts.json").write_text("canon\n", encoding="utf-8")
    via = work / "via-link"
    via.symlink_to(nested / "facts.json")
    (work / "other.txt").write_text("other\n", encoding="utf-8")
    bare = tmp_path / "bare"
    bare.mkdir()
    unguarded = _bash(SPLICE, bare)
    assert unguarded.returncode == 0 and (bare / "facts.json").read_text(encoding="utf-8") == "x"

    wrapped = guard_command(SPLICE, env_type="local", home=str(home))
    assert isinstance(wrapped, str) and wrapped != SPLICE
    guarded = _bash(_session(wrapped, work), work)
    assert not (work / "facts.json").read_text(encoding="utf-8") == "x"
    assert facts.read_text(encoding="utf-8") == "old-facts\n"

    cases = {
        "python": "python3 -c " + shlex.quote(
            'name="facts"; ext="json"; open(name+"."+ext,"w").write("z")'
        ),
        "child": "python3 -c " + shlex.quote(
            'import subprocess; subprocess.run(["/bin/bash","-c",'
            '"name=facts; ext=json; printf child > ${name}.${ext}"])'
        ),
        "append": "printf more >> \"$(printf '%s.%s' facts json)\"",
        "remove": "rm -f \"$(printf '%s.%s' facts json)\"",
        "rename": "mv other.txt \"$(printf '%s.%s' facts json)\"",
        "link": "ln -s other.txt \"$(printf '%s.%s' trade_plan json)\"",
        "through": "printf hijack > via-link",
        "directory": "mkdir \"$(printf '%s.%s' bundle json)\"",
    }
    for name, command in cases.items():
        assert "facts.json" not in command and "trade_plan.json" not in command and "bundle.json" not in command
        result = _bash(guard_command(command, env_type="local", home=str(home)), work)
        assert facts.read_text(encoding="utf-8") == "old-facts\n", name
        assert (nested / "facts.json").read_text(encoding="utf-8") == "canon\n", name
        if name not in {"python", "child"}:
            assert result.returncode != 0, (name, result.returncode, result.stderr)
    assert not (work / "trade_plan.json").exists()
    assert not (work / "bundle.json").exists()
    assert via.is_symlink()
    notes = guard_command("printf notes-ok > notes.txt && printf report-ok > report.md", env_type="local", home=str(home))
    assert _bash(notes, work).returncode == 0
    assert (work / "notes.txt").read_text(encoding="utf-8") == "notes-ok"
    assert (work / "report.md").read_text(encoding="utf-8") == "report-ok"
    assert _bash(guard_command("echo hi", env_type="local", home=str(home)), work).stdout.strip() == "hi"
    from tools.authority_os_guard import darwin_spawn_prefix
    prefix = darwin_spawn_prefix(frozenset(NAMES))
    direct = subprocess.run(
        prefix + [sys.executable, "-c", "name='facts'; ext='json'; open(name+'.'+ext,'w').write('z')"],
        cwd=work, capture_output=True, text=True,
    )
    assert direct.returncode != 0
    assert facts.read_text(encoding="utf-8") == "old-facts\n"


def test_writable_writer_is_not_executed_and_immutable_writer_is(tmp_path):
    home = _home(tmp_path)
    work = tmp_path / "run"
    work.mkdir()
    script = tmp_path / "authority_write.py"
    script.write_text(textwrap.dedent("""\
        import pathlib, sys
        args = sys.argv
        target = pathlib.Path(args[args.index("--run-dir") + 1]) / args[args.index("--relative") + 1]
        target.write_text('{"marker":"legal"}\\n', encoding="utf-8")
        print("wrote")
    """), encoding="utf-8")
    register_official_writer(str(script))
    command = (
        f"python3 {shlex.quote(str(script))} --kind run-card --run-dir {shlex.quote(str(work))} "
        f"--relative run-card.json --input {shlex.quote(str(work / 'in.json'))}"
    )
    # shlex.quote inside the command would add quotes the argv parser accepts.
    command = (
        f"python3 {script} --kind run-card --run-dir {work} "
        "--relative run-card.json --input " + str(work / "in.json")
    )
    assert official_exec_argv(command) is not None
    writable = guard_command(command, env_type="local", home=str(home))
    refused = _bash(writable, work)
    assert refused.returncode == 126
    assert "writable" in refused.stderr
    assert not (work / "run-card.json").exists()
    script.chmod(script.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
    allowed = _bash(guard_command(command, env_type="local", home=str(home)), work)
    assert allowed.returncode == 0, allowed.stderr
    assert json.loads((work / "run-card.json").read_text(encoding="utf-8")) == {"marker": "legal"}


def test_darwin_spawn_prefix_keeps_an_inherited_pipe(tmp_path):
    if sys.platform != "darwin":
        pytest.skip("seatbelt runs on the host")
    from tools.authority_os_guard import darwin_spawn_prefix
    prefix = darwin_spawn_prefix(frozenset(NAMES))
    assert prefix
    read_fd, write_fd = os.pipe()
    try:
        proc = subprocess.Popen(
            prefix + [sys.executable, "-c", "import os; os.read(int(os.environ['FD']), 1)"],
            cwd=tmp_path,
            env={**os.environ, "FD": str(read_fd)},
            pass_fds=(read_fd,),
            close_fds=True,
        )
        os.close(read_fd)
        read_fd = -1
        os.write(write_fd, b"Z")
        os.close(write_fd)
        write_fd = -1
        assert proc.wait(timeout=15) == 0
    finally:
        if read_fd >= 0:
            os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)


def _container() -> str:
    found = subprocess.run(
        ["docker", "ps", "--filter", f"name={CONTAINER}", "--format", "{{.ID}}"],
        capture_output=True, text=True, check=False,
    )
    rows = [line.strip() for line in found.stdout.splitlines() if line.strip()]
    if found.returncode != 0 or not rows:
        pytest.fail(f"Big container {CONTAINER} is not running")
    return rows[0]


def _docker(cid: str, script: str, workdir: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "exec", "-w", workdir, cid, "bash", "-c", script],
        capture_output=True, text=True, timeout=120,
    )


class _ContainerEnv:
    """Minimal terminal env: same session quoting the Docker backend applies."""

    def __init__(self, cid: str, work: str):
        self.cid = cid
        self.cwd = work
        self.env_type = "docker"
        self.timeout = 60

    def execute(self, command, cwd="", timeout=None, stdin_data=None, **kwargs):
        from tools.environments.base_session_env import _split_cwd_marker

        workdir = cwd or self.cwd
        script = _session(command, Path(workdir))
        proc = subprocess.run(
            ["docker", "exec", "-i", "-w", workdir, self.cid, "bash", "-c", script],
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = proc.stdout
        split = _split_cwd_marker(output, "@@HERMES_CWD@@")
        if split is not None:
            output = split[1]
        return {"output": output, "returncode": proc.returncode}


def _assert_file_ops_cannot_replace_authority(cid: str, work: str, home: Path):
    """write/delete/move go through the shell file ops, not only the terminal tool."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.file_operations import ShellFileOperations

    token = set_hermes_home_override(home)
    try:
        ops = ShellFileOperations(_ContainerEnv(cid, work))
        denied = ops.write_file(f"{work}/facts.json", '{"marker":"nope"}\n')
        assert denied.error, "direct file-op write was not refused"
        link = ops.write_file(f"{work}/via-link", '{"marker":"nope"}\n')
        assert link.error, "symlink file-op write was not refused"
        removed = ops.delete_file(f"{work}/facts.json")
        assert removed.error, "file-op delete was not refused"
        renamed = ops.move_file(f"{work}/nest", f"{work}/nest-away")
        assert renamed.error, "directory move was not refused"
        plain = ops.write_file(f"{work}/plain.txt", "plain-ok\n")
        assert plain.error is None, plain.error
    finally:
        reset_hermes_home_override(token)
    assert _docker(cid, "cat facts.json", work).stdout == "old-facts\n"
    assert _docker(cid, "cat sub/facts.json", work).stdout == "canon\n"
    assert _docker(cid, "cat nest/facts.json", work).stdout == "nested\n"
    assert _docker(cid, "cat plain.txt", work).stdout == "plain-ok\n"
    assert _docker(cid, "test ! -e nest-away", work).returncode == 0


def test_big_container_blocks_the_splice_and_still_runs_the_writer(tmp_path):
    home = _home(tmp_path)
    cid = _container()
    work = f"/tmp/hermes-os-guard-{uuid.uuid4().hex}"
    setup = _docker(
        cid,
        "mkdir -p "
        + shlex.quote(work + "/sub")
        + " "
        + shlex.quote(work + "/nest")
        + " && printf 'old-facts\\n' > "
        + shlex.quote(work + "/facts.json")
        + " && printf 'canon\\n' > "
        + shlex.quote(work + "/sub/facts.json")
        + " && printf 'nested\\n' > "
        + shlex.quote(work + "/nest/facts.json")
        + " && ln -s sub/facts.json "
        + shlex.quote(work + "/via-link")
        + " && printf other > "
        + shlex.quote(work + "/other.txt")
        + " && printf '%s' '{\"marker\":\"legal\"}' > "
        + shlex.quote(work + "/in.json"),
        "/tmp",
    )
    assert setup.returncode == 0, setup.stderr
    try:
        wrapped = guard_command(SPLICE, env_type="docker", home=str(home))
        assert isinstance(wrapped, str) and wrapped != SPLICE and "facts.json" not in SPLICE
        splice = _docker(cid, _session(wrapped, Path(work)), work)
        body = _docker(cid, "cat facts.json", work)
        assert body.stdout == "old-facts\n", splice.stderr
        assert splice.returncode != 0
        script = "\n".join((
            "set +e",
            "printf notes-ok > notes.txt",
            "python3 -c " + shlex.quote('name="facts"; ext="json"; open(name+"."+ext,"w").write("z")'),
            "python3 -c " + shlex.quote(
                'import subprocess; subprocess.run(["/bin/bash","-c","name=facts; ext=json; printf child > ${name}.${ext}"])'
            ),
            "python3 -c " + shlex.quote(
                'import ctypes,sys; libc=ctypes.CDLL(None); libc.syscall.restype=ctypes.c_long; '
                'rc=libc.syscall(56,-100,b"facts"+b"."+b"json",577,420); sys.exit(0 if rc>=0 else 1)'
            ),
            "printf more >> \"$(printf '%s.%s' facts json)\"",
            "rm -f \"$(printf '%s.%s' facts json)\"",
            "mv other.txt \"$(printf '%s.%s' facts json)\"",
            "ln -s notes.txt \"$(printf '%s.%s' trade_plan json)\"",
            "mkdir \"$(printf '%s.%s' bundle json)\"",
            "printf hijack > via-link",
            "cat \"$(printf '%s.%s' facts json)\"",
            "echo hi",
        ))
        for line in script.splitlines():
            assert "facts.json" not in line and "trade_plan.json" not in line and "bundle.json" not in line
        ran = _docker(cid, _session(guard_command(script, env_type="docker", home=str(home)), Path(work)), work)
        assert "old-facts" in ran.stdout
        assert "\nhi\n" in f"\n{ran.stdout}"
        after = _docker(
            cid,
            "python3 -c \"import pathlib; p=pathlib.Path('.'); "
            "print(repr((p/'facts.json').read_bytes())); "
            "print(repr((p/'sub'/'facts.json').read_bytes())); "
            "print(repr((p/'notes.txt').read_bytes())); "
            "print((p/'trade_plan.json').exists()); print((p/'bundle.json').exists())\"",
            work,
        )
        assert after.stdout == "b'old-facts\\n'\nb'canon\\n'\nb'notes-ok'\nFalse\nFalse\n", after.stdout + after.stderr
        touch = _docker(cid, "touch /root/.hermes/skills/finance/trading-agents/scripts/authority_write.py", work)
        assert touch.returncode != 0
        writer = "/root/.hermes/skills/finance/trading-agents/scripts/authority_write.py"
        register_official_writer(writer)
        legal = (
            f"python3 {writer} --kind run-card --run-dir {work} "
            f"--relative run-card.json --input {work}/in.json"
        )
        assert official_exec_argv(legal) is not None
        wrote = _docker(cid, _session(guard_command(legal, env_type="docker", home=str(home)), Path(work)), work)
        assert wrote.returncode == 0, wrote.stderr
        card = _docker(cid, "cat run-card.json", work)
        assert json.loads(card.stdout)["marker"] == "legal"
        mixed = (
            "f=$(printf '%s.%s' run-card json); "
            f"python3 {writer} --kind run-card --run-dir \"$PWD\" --relative \"$f\" --input \"$PWD/in.json\"; "
            "name=facts; ext=json; printf x > ${name}.${ext}"
        )
        assert official_exec_argv(mixed) is None
        assert "facts.json" not in mixed and "run-card.json" not in mixed
        _docker(cid, _session(guard_command(mixed, env_type="docker", home=str(home)), Path(work)), work)
        mixed_body = _docker(cid, "cat facts.json", work)
        assert mixed_body.stdout == "old-facts\n"
        from tools.authority_os_guard import linux_argv_command
        argv_code = (
            "name='facts'; ext='json'\n"
            "try:\n"
            "    open(name + '.' + ext, 'w').write('nope')\n"
            "    raise SystemExit('wrote')\n"
            "except PermissionError:\n"
            "    pass\n"
            "open('argv-notes.txt','w').write('argv-ok')\n"
        )
        argv_run = _docker(
            cid,
            _session(linux_argv_command(["python3", "-c", argv_code], NAMES), Path(work)),
            work,
        )
        assert argv_run.returncode == 0, argv_run.stderr
        argv_notes = _docker(cid, "cat argv-notes.txt", work)
        assert argv_notes.stdout == "argv-ok"
        assert _docker(cid, "cat facts.json", work).stdout == "old-facts\n"
        moved = _docker(
            cid,
            _session(guard_command("mv nest nest-away", env_type="docker", home=str(home)), Path(work)),
            work,
        )
        assert moved.returncode != 0, moved.stderr
        assert _docker(cid, "cat nest/facts.json", work).stdout == "nested\n"
        assert _docker(cid, "test ! -e nest-away", work).returncode == 0
        _assert_file_ops_cannot_replace_authority(cid, work, home)
    finally:
        _docker(cid, f"rm -rf {shlex.quote(work)}", "/tmp")


def test_darwin_blocks_hardlink_rename_and_untrusted_interpreter(tmp_path):
    if sys.platform != "darwin":
        pytest.skip("seatbelt runs on the host")
    home = _home(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    facts = work / "facts.json"
    facts.write_text("old-facts\n", encoding="utf-8")
    os.link(facts, work / "alias.json")
    nest = work / "nest"
    nest.mkdir()
    (nest / "facts.json").write_text("nested\n", encoding="utf-8")
    empty = work / "empty"
    empty.mkdir()
    (empty / "plain.txt").write_text("plain\n", encoding="utf-8")
    (work / "notes.txt").write_text("notes\n", encoding="utf-8")
    os.link(work / "notes.txt", work / "notes-link")

    alias = _bash(guard_command("printf HL > alias.json", env_type="local", home=str(home)), work)
    assert alias.returncode != 0, alias.stderr
    assert facts.read_bytes() == b"old-facts\n"
    assert (work / "alias.json").read_bytes() == b"old-facts\n"

    moved = _bash(guard_command("mv nest nest-away", env_type="local", home=str(home)), work)
    assert moved.returncode != 0, moved.stderr
    assert (nest / "facts.json").read_bytes() == b"nested\n"
    assert not (work / "nest-away").exists()

    ordinary = _bash(
        guard_command(
            "mv empty empty-away && printf link-ok > notes-link && printf still-ok > plain.txt",
            env_type="local",
            home=str(home),
        ),
        work,
    )
    assert ordinary.returncode == 0, ordinary.stderr
    assert (work / "empty-away" / "plain.txt").read_text(encoding="utf-8") == "plain\n"
    assert (work / "notes.txt").read_text(encoding="utf-8") == "link-ok"
    assert (work / "plain.txt").read_text(encoding="utf-8") == "still-ok"

    script = tmp_path / "authority_write.py"
    script.write_text(textwrap.dedent("""\
        import pathlib, sys
        args = sys.argv
        target = pathlib.Path(args[args.index("--run-dir") + 1]) / args[args.index("--relative") + 1]
        target.write_text('{"marker":"legal"}\\n', encoding="utf-8")
    """), encoding="utf-8")
    script.chmod(0o444)
    register_official_writer(str(script))
    fake = tmp_path / "python3"
    fake.write_text(
        "#!/bin/sh\nprintf pwned > escaped.txt\nprintf pwned > facts.json\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    fake_cmd = (
        f"{fake} {script} --kind run-card --run-dir {work} "
        "--relative notes.json --input " + str(work / "in.json")
    )
    assert official_exec_argv(fake_cmd) is not None
    refused = _bash(guard_command(fake_cmd, env_type="local", home=str(home)), work)
    assert refused.returncode == 126, refused.stderr
    assert not (work / "escaped.txt").exists()
    assert facts.read_bytes() == b"old-facts\n"

    user_home = tmp_path / "userhome"
    probe = subprocess.run(
        ["/usr/bin/python3", "-c", "import site; print(site.getusersitepackages())"],
        cwd=work,
        capture_output=True,
        text=True,
        env={**os.environ, "HOME": str(user_home)},
        check=False,
    )
    assert probe.returncode == 0, probe.stderr
    site_dir = Path(probe.stdout.strip())
    site_dir.mkdir(parents=True)
    (site_dir / "usercustomize.py").write_text(
        "open('facts.json','w').write('pwned-user' + chr(10))\n",
        encoding="utf-8",
    )
    bare = subprocess.run(
        ["/usr/bin/python3", "-c", "print('site-loaded')"],
        cwd=work,
        capture_output=True,
        text=True,
        env={**os.environ, "HOME": str(user_home)},
    )
    assert bare.returncode == 0, bare.stderr
    assert facts.read_bytes() == b"pwned-user\n"
    facts.write_text("old-facts\n", encoding="utf-8")
    legal = (
        f"python3 {script} --kind run-card --run-dir {work} "
        "--relative run-card.json --input " + str(work / "in.json")
    )
    allowed = subprocess.run(
        ["/bin/bash", "-c", guard_command(legal, env_type="local", home=str(home))],
        cwd=work,
        capture_output=True,
        text=True,
        env={**os.environ, "HOME": str(user_home)},
    )
    assert allowed.returncode == 0, allowed.stderr
    assert facts.read_bytes() == b"old-facts\n"
    assert json.loads((work / "run-card.json").read_text(encoding="utf-8")) == {"marker": "legal"}
    outside = tmp_path / "outside"
    outside.mkdir()
    os.link(facts, outside / "alias.json")
    external = _bash(
        guard_command(
            "printf HL > " + shlex.quote(str(outside / "alias.json")),
            env_type="local",
            home=str(home),
        ),
        work,
    )
    assert external.returncode != 0, external.stderr
    assert facts.read_bytes() == b"old-facts\n"
    assert (outside / "alias.json").read_bytes() == b"old-facts\n"
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    hidden_facts = hidden / "facts.json"
    hidden_facts.write_text("old-facts\n", encoding="utf-8")
    os.link(hidden_facts, hidden / "alias.json")
    empty = tmp_path / "empty-cwd"
    empty.mkdir()
    hidden_write = _bash(
        guard_command(
            "printf EMPTY > " + shlex.quote(str(hidden / "alias.json")),
            env_type="local",
            home=str(home),
        ),
        empty,
    )
    assert hidden_write.returncode != 0, hidden_write.stderr
    assert hidden_facts.read_bytes() == b"old-facts\n"
    assert (hidden / "alias.json").read_bytes() == b"old-facts\n"
    ordinary_there = _bash(
        guard_command("printf still-there > local-notes.txt", env_type="local", home=str(home)),
        empty,
    )
    assert ordinary_there.returncode == 0, ordinary_there.stderr
    assert (empty / "local-notes.txt").read_text(encoding="utf-8") == "still-there"


def test_big_container_blocks_hardlink_procfd_and_untrusted_writer(tmp_path):
    home = _home(tmp_path)
    cid = _container()
    work = f"/tmp/hermes-os-guard-bypass-{uuid.uuid4().hex}"
    setup = _docker(
        cid,
        "mkdir -p " + shlex.quote(work)
        + " && printf 'old-facts\\n' > " + shlex.quote(work + "/facts.json")
        + " && ln " + shlex.quote(work + "/facts.json") + " " + shlex.quote(work + "/alias.json")
        + " && printf notes > " + shlex.quote(work + "/notes.txt")
        + " && ln " + shlex.quote(work + "/notes.txt") + " " + shlex.quote(work + "/notes-link")
        + " && printf '%s' '{\"marker\":\"legal\"}' > " + shlex.quote(work + "/in.json"),
        "/tmp",
    )
    assert setup.returncode == 0, setup.stderr
    try:
        hard = _docker(
            cid,
            _session(guard_command("printf hardlink-write > alias.json", env_type="docker", home=str(home)), Path(work)),
            work,
        )
        assert hard.returncode != 0, hard.stderr
        assert _docker(cid, "cat facts.json", work).stdout == "old-facts\n"
        assert _docker(cid, "cat alias.json", work).stdout == "old-facts\n"
        outside = "/tmp/hermes-os-guard-outside-" + uuid.uuid4().hex
        linked = _docker(
            cid,
            "mkdir -p " + shlex.quote(outside)
            + " && ln facts.json " + shlex.quote(outside + "/alias.json"),
            work,
        )
        assert linked.returncode == 0, linked.stderr
        external = _docker(
            cid,
            _session(
                guard_command(
                    "printf HL > " + shlex.quote(outside + "/alias.json"),
                    env_type="docker",
                    home=str(home),
                ),
                Path(work),
            ),
            work,
        )
        assert external.returncode != 0, external.stderr
        assert _docker(cid, "cat facts.json", work).stdout == "old-facts\n"
        assert _docker(cid, "cat " + shlex.quote(outside + "/alias.json"), "/").stdout == "old-facts\n"
        _docker(cid, "rm -rf " + shlex.quote(outside), "/tmp")
        ordinary = _docker(
            cid,
            _session(
                guard_command("printf link-ok > notes-link && printf still-ok > plain.txt", env_type="docker", home=str(home)),
                Path(work),
            ),
            work,
        )
        assert ordinary.returncode == 0, ordinary.stderr
        assert _docker(cid, "cat notes.txt", work).stdout == "link-ok"
        proc = (
            "python3 -c " + shlex.quote(
                "import os\n"
                "name = 'facts'\n"
                "ext = 'json'\n"
                "fd = os.open(name + '.' + ext, os.O_RDONLY)\n"
                "out = os.open('/proc/self/fd/%d' % fd, os.O_WRONLY | os.O_TRUNC)\n"
                "os.write(out, b'procfd')\n"
            )
        )
        shell_fd = "exec 3< \"$(printf '%s.%s' facts json)\"; printf shellfd > /proc/self/fd/3"
        assert "facts.json" not in proc and "facts.json" not in shell_fd
        dev_py = (
            "python3 -c " + shlex.quote(
                "import os\n"
                "name = 'facts'\n"
                "ext = 'json'\n"
                "fd = os.open(name + '.' + ext, os.O_RDONLY)\n"
                "out = os.open('/dev/fd/%d' % fd, os.O_WRONLY | os.O_TRUNC)\n"
                "os.write(out, b'devfd')\n"
            )
        )
        dev_sh = "exec 3< \"$(printf '%s.%s' facts json)\"; printf devsh > /dev/fd/3"
        dev_in = "exec < \"$(printf '%s.%s' facts json)\"; printf devin > /dev/stdin"
        root_fd = (
            "python3 -c " + shlex.quote(
                "import os\n"
                "name = 'facts'\n"
                "ext = 'json'\n"
                "fd = os.open(name + '.' + ext, os.O_RDONLY)\n"
                "out = os.open('/proc/self/root/dev/fd/%d' % fd, os.O_WRONLY | os.O_TRUNC)\n"
                "os.write(out, b'rootfd')\n"
            )
        )
        nested_fd = (
            "python3 -c " + shlex.quote(
                "import os\n"
                "name = 'facts'\n"
                "ext = 'json'\n"
                "fd = os.open(name + '.' + ext, os.O_RDONLY)\n"
                "out = os.open('/proc/self/root/proc/self/fd/%d' % fd, os.O_WRONLY | os.O_TRUNC)\n"
                "os.write(out, b'nestedfd')\n"
            )
        )
        deep_fd = (
            "python3 -c " + shlex.quote(
                "import os\n"
                "name = 'facts'\n"
                "ext = 'json'\n"
                "fd = os.open(name + '.' + ext, os.O_RDONLY)\n"
                "opened = '/proc/self/root/proc/self/root/dev/fd/%d' % fd\n"
                "out = os.open(opened, os.O_WRONLY | os.O_TRUNC)\n"
                "os.write(out, b'deepfd')\n"
            )
        )
        task_fd = (
            "python3 -c " + shlex.quote(
                "import os\n"
                "name = 'facts'\n"
                "ext = 'json'\n"
                "fd = os.open(name + '.' + ext, os.O_RDONLY)\n"
                "opened = '/proc/self/task/%d/fd/%d' % (os.getpid(), fd)\n"
                "out = os.open(opened, os.O_WRONLY | os.O_TRUNC)\n"
                "os.write(out, b'taskfd')\n"
            )
        )
        thread_fd = (
            "python3 -c " + shlex.quote(
                "import os\n"
                "name = 'facts'\n"
                "ext = 'json'\n"
                "fd = os.open(name + '.' + ext, os.O_RDONLY)\n"
                "out = os.open('/proc/thread-self/fd/%d' % fd, os.O_WRONLY | os.O_TRUNC)\n"
                "os.write(out, b'threadfd')\n"
            )
        )
        pid_fd = (
            "python3 -c " + shlex.quote(
                "import os\n"
                "name = 'facts'\n"
                "ext = 'json'\n"
                "fd = os.open(name + '.' + ext, os.O_RDONLY)\n"
                "opened = '/proc/%d/root/dev/fd/%d' % (os.getpid(), fd)\n"
                "out = os.open(opened, os.O_WRONLY | os.O_TRUNC)\n"
                "os.write(out, b'pidfd')\n"
            )
        )
        dir_fd = (
            "python3 -c " + shlex.quote(
                "import os\n"
                "name = 'facts'\n"
                "ext = 'json'\n"
                "fd = os.open(name + '.' + ext, os.O_RDONLY)\n"
                "folder = os.open('/proc/self/fd', os.O_RDONLY)\n"
                "out = os.open('%d' % fd, os.O_WRONLY | os.O_TRUNC, dir_fd=folder)\n"
                "os.write(out, b'dirfd')\n"
            )
        )
        from_root = (
            "python3 -c " + shlex.quote(
                "import os\n"
                "name = 'facts'\n"
                "ext = 'json'\n"
                "fd = os.open(name + '.' + ext, os.O_RDONLY)\n"
                "os.dup2(fd, 3)\n"
                "os.chdir('/')\n"
                "out = os.open('proc/self/root/dev/fd/3', os.O_WRONLY | os.O_TRUNC)\n"
                "os.write(out, b'fromroot')\n"
            )
        )
        fd_link = "/tmp/hermes-os-fd-link-" + uuid.uuid4().hex
        other_dir = "/tmp/hermes-os-other-" + uuid.uuid4().hex
        planted_link = _docker(
            cid,
            "ln -s /proc/self/root/dev/fd/3 magic-link"
            " && mkdir -p " + shlex.quote(other_dir)
            + " && ln -s " + shlex.quote(other_dir) + " jump"
            + " && ln -s /proc/self/root/dev/fd/3 " + shlex.quote(fd_link),
            work,
        )
        assert planted_link.returncode == 0, planted_link.stderr
        magic = (
            "python3 -c " + shlex.quote(
                "import os\n"
                "name = 'facts'\n"
                "ext = 'json'\n"
                "fd = os.open(name + '.' + ext, os.O_RDONLY)\n"
                "os.dup2(fd, 3)\n"
                "out = os.open('magic-link', os.O_WRONLY | os.O_TRUNC)\n"
                "os.write(out, b'magiclink')\n"
            )
        )
        dotdot = (
            "python3 -c " + shlex.quote(
                "import os\n"
                "name = 'facts'\n"
                "ext = 'json'\n"
                "fd = os.open(name + '.' + ext, os.O_RDONLY)\n"
                "os.dup2(fd, 3)\n"
                "out = os.open('jump/../" + os.path.basename(fd_link) + "', os.O_WRONLY | os.O_TRUNC)\n"
                "os.write(out, b'dotdot')\n"
            )
        )
        map_fd = (
            "python3 -c " + shlex.quote(
                "import os, mmap\n"
                "name = 'facts'\n"
                "ext = 'json'\n"
                "target = name + '.' + ext\n"
                "fd = os.open(target, os.O_RDONLY)\n"
                "mapped = mmap.mmap(fd, os.fstat(fd).st_size, prot=mmap.PROT_READ)\n"
                "want = os.path.realpath(target)\n"
                "found = ''\n"
                "for line in open('/proc/self/maps'):\n"
                "    parts = line.split()\n"
                "    if parts and (parts[-1] == want or parts[-1].endswith('/' + target)):\n"
                "        found = parts[0]\n"
                "        break\n"
                "if not found:\n"
                "    raise SystemExit(2)\n"
                "open('map-tried', 'w').write('1')\n"
                "try:\n"
                "    out = os.open('/proc/self/map_files/' + found, os.O_WRONLY | os.O_TRUNC)\n"
                "    os.write(out, b'mapfiles')\n"
                "    open('map-result', 'w').write('wrote')\n"
                "except OSError as exc:\n"
                "    open('map-result', 'w').write('err %s' % exc.errno)\n"
            )
        )
        empty_link = (
            "python3 -c " + shlex.quote(
                "import ctypes, os\n"
                "name = 'facts'\n"
                "ext = 'json'\n"
                "fd = os.open(name + '.' + ext, os.O_RDONLY)\n"
                "libc = ctypes.CDLL(None, use_errno=True)\n"
                "libc.syscall.restype = ctypes.c_long\n"
                "rc = libc.syscall(37, fd, b'', -100, b'empty-alias.json', 0x1000)\n"
                "open('empty-status', 'w').write('%s %s' % (rc, ctypes.get_errno()))\n"
            )
        )
        commands = (
            proc, shell_fd, dev_py, dev_sh, dev_in, root_fd, nested_fd, deep_fd,
            task_fd, thread_fd, from_root, pid_fd, dir_fd, magic, dotdot, map_fd, empty_link,
        )
        for command in commands:
            assert "facts.json" not in command
            ran = _docker(
                cid,
                _session(guard_command(command, env_type="docker", home=str(home)), Path(work)),
                work,
            )
            assert _docker(cid, "cat facts.json", work).stdout == "old-facts\n", (command, ran.returncode, ran.stderr)
        assert _docker(cid, "test -s map-tried", work).returncode == 0
        assert _docker(cid, "cat map-result", work).stdout == "err 13"
        planted_root = _docker(
            cid,
            "mkdir -p proc/self/fd hgbox"
            " && printf 'decoy\\n' > decoy.txt"
            " && ln \"$(printf '%s.%s' facts json)\" proc/self/fd/3"
            " && ln \"$(printf '%s.%s' facts json)\" hgbox/alias",
            work,
        )
        assert planted_root.returncode == 0, planted_root.stderr
        in_root = (
            "python3 -c " + shlex.quote(
                "import ctypes, os\n"
                "libc = ctypes.CDLL(None, use_errno=True)\n"
                "libc.syscall.restype = ctypes.c_long\n"
                "libc.syscall.argtypes = [ctypes.c_long, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]\n"
                "class How(ctypes.Structure):\n"
                "    _fields_ = (('flags', ctypes.c_uint64), ('mode', ctypes.c_uint64), ('resolve', ctypes.c_uint64))\n"
                "def u64(value):\n"
                "    return value & ((1 << 64) - 1)\n"
                "fd = os.open('decoy.txt', os.O_RDONLY)\n"
                "os.dup2(fd, 3)\n"
                "how = How(os.O_WRONLY | os.O_TRUNC, 0, 0x10)\n"
                "path = ctypes.create_string_buffer(b'/proc/self/fd/3')\n"
                "rc = libc.syscall(437, u64(-100), ctypes.addressof(path), ctypes.addressof(how), 24, 0)\n"
                "if rc >= 0:\n"
                "    os.write(rc, b'planted\\n')\n"
            )
        )
        alias_root = (
            "python3 -c " + shlex.quote(
                "import ctypes, os\n"
                "libc = ctypes.CDLL(None, use_errno=True)\n"
                "libc.syscall.restype = ctypes.c_long\n"
                "libc.syscall.argtypes = [ctypes.c_long, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]\n"
                "class How(ctypes.Structure):\n"
                "    _fields_ = (('flags', ctypes.c_uint64), ('mode', ctypes.c_uint64), ('resolve', ctypes.c_uint64))\n"
                "def u64(value):\n"
                "    return value & ((1 << 64) - 1)\n"
                "how = How(os.O_WRONLY | os.O_TRUNC, 0, 0x10)\n"
                "path = ctypes.create_string_buffer(b'/hgbox/alias')\n"
                "rc = libc.syscall(437, u64(-100), ctypes.addressof(path), ctypes.addressof(how), 24, 0)\n"
                "if rc >= 0:\n"
                "    os.write(rc, b'aliasbox\\n')\n"
            )
        )
        plain_root = (
            "python3 -c " + shlex.quote(
                "import ctypes, os\n"
                "libc = ctypes.CDLL(None, use_errno=True)\n"
                "libc.syscall.restype = ctypes.c_long\n"
                "libc.syscall.argtypes = [ctypes.c_long, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]\n"
                "class How(ctypes.Structure):\n"
                "    _fields_ = (('flags', ctypes.c_uint64), ('mode', ctypes.c_uint64), ('resolve', ctypes.c_uint64))\n"
                "def u64(value):\n"
                "    return value & ((1 << 64) - 1)\n"
                "how = How(os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644, 0x10)\n"
                "path = ctypes.create_string_buffer(b'/plain-openat2.txt')\n"
                "rc = libc.syscall(437, u64(-100), ctypes.addressof(path), ctypes.addressof(how), 24, 0)\n"
                "if rc >= 0:\n"
                "    os.write(rc, b'plain-ok')\n"
            )
        )
        for command in (in_root, alias_root, plain_root):
            assert "facts.json" not in command
            ran = _docker(
                cid,
                _session(guard_command(command, env_type="docker", home=str(home)), Path(work)),
                work,
            )
            assert _docker(cid, "cat facts.json", work).stdout == "old-facts\n", (command, ran.returncode, ran.stderr)
        assert _docker(cid, "cat decoy.txt", work).stdout == "decoy\n"
        assert _docker(cid, "cat plain-openat2.txt", work).stdout == "plain-ok"
        wide_how = (
            "python3 -c " + shlex.quote(
                "import ctypes, os\n"
                "libc = ctypes.CDLL(None, use_errno=True)\n"
                "libc.syscall.restype = ctypes.c_long\n"
                "libc.syscall.argtypes = [ctypes.c_long, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]\n"
                "def u64(value):\n"
                "    return value & ((1 << 64) - 1)\n"
                "name = b'facts'\n"
                "ext = b'json'\n"
                "size = 257\n"
                "buf = ctypes.create_string_buffer(size)\n"
                "raw = (ctypes.c_uint64 * 3)(os.O_WRONLY | os.O_TRUNC, 0, 0)\n"
                "ctypes.memmove(buf, raw, 24)\n"
                "path = ctypes.create_string_buffer(name + b'.' + ext)\n"
                "rc = libc.syscall(437, u64(-100), ctypes.addressof(path), ctypes.addressof(buf), u64(size), 0)\n"
                "if rc >= 0:\n"
                "    os.write(rc, b'size257\\n')\n"
            )
        )
        wide_root = (
            "python3 -c " + shlex.quote(
                "import ctypes, os\n"
                "libc = ctypes.CDLL(None, use_errno=True)\n"
                "libc.syscall.restype = ctypes.c_long\n"
                "libc.syscall.argtypes = [ctypes.c_long, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]\n"
                "def u64(value):\n"
                "    return value & ((1 << 64) - 1)\n"
                "fd = os.open('decoy.txt', os.O_RDONLY)\n"
                "os.dup2(fd, 3)\n"
                "size = 4096\n"
                "buf = ctypes.create_string_buffer(size)\n"
                "raw = (ctypes.c_uint64 * 3)(os.O_WRONLY | os.O_TRUNC, 0, 0x10)\n"
                "ctypes.memmove(buf, raw, 24)\n"
                "path = ctypes.create_string_buffer(b'/proc/self/fd/3')\n"
                "rc = libc.syscall(437, u64(-100), ctypes.addressof(path), ctypes.addressof(buf), u64(size), 0)\n"
                "if rc >= 0:\n"
                "    os.write(rc, b'size4096\\n')\n"
            )
        )
        for command in (wide_how, wide_root):
            assert "facts.json" not in command
            ran = _docker(
                cid,
                _session(guard_command(command, env_type="docker", home=str(home)), Path(work)),
                work,
            )
            assert _docker(cid, "cat facts.json", work).stdout == "old-facts\n", (command, ran.returncode, ran.stderr)
        assert _docker(cid, "cat decoy.txt", work).stdout == "decoy\n"
        assert _docker(cid, "test ! -e empty-alias.json", work).returncode == 0
        status = _docker(cid, "cat empty-status", work)
        assert status.stdout.startswith("-1 "), status.stdout
        writer = "/root/.hermes/skills/finance/trading-agents/scripts/authority_write.py"
        register_official_writer(writer)
        planted = _docker(
            cid,
            "mkdir -p /tmp/fake-py && printf '%s\\n' '#!/bin/sh' 'printf pwned > escaped.txt' 'printf pwned > facts.json' > /tmp/fake-py/python3 && chmod 755 /tmp/fake-py/python3",
            work,
        )
        assert planted.returncode == 0, planted.stderr
        fake_cmd = (
            f"/tmp/fake-py/python3 {writer} --kind run-card --run-dir {work} "
            f"--relative notes.json --input {work}/in.json"
        )
        assert official_exec_argv(fake_cmd) is not None
        fake_run = _docker(
            cid,
            _session(guard_command(fake_cmd, env_type="docker", home=str(home)), Path(work)),
            work,
        )
        assert fake_run.returncode == 126, fake_run.stderr
        assert _docker(cid, "cat facts.json", work).stdout == "old-facts\n"
        assert _docker(cid, "test ! -e escaped.txt", work).returncode == 0
        site = _docker(
            cid,
            "/usr/local/bin/python3 -c 'import os,site; p=site.getusersitepackages(); os.makedirs(p, exist_ok=True); print(p)'",
            work,
        )
        assert site.returncode == 0, site.stderr
        site_dir = site.stdout.strip().splitlines()[-1]
        payload = "open('facts.json','w').write('pwned-user' + chr(10))\n"
        plant_user = _docker(
            cid,
            "mkdir -p " + shlex.quote(site_dir)
            + " && printf '%s' " + shlex.quote(payload)
            + " > " + shlex.quote(site_dir + "/usercustomize.py"),
            work,
        )
        if plant_user.returncode != 0:
            plant_user = subprocess.run(
                ["docker", "exec", "-u", "0", "-w", work, cid, "bash", "-c",
                 "mkdir -p " + shlex.quote(site_dir)
                 + " && printf '%s' " + shlex.quote(payload)
                 + " > " + shlex.quote(site_dir + "/usercustomize.py")
                 + " && chmod 755 " + shlex.quote(site_dir)
                 + " && chmod 644 " + shlex.quote(site_dir + "/usercustomize.py")],
                capture_output=True, text=True, timeout=60,
            )
        assert plant_user.returncode == 0, plant_user.stderr
        control = _docker(cid, "/usr/local/bin/python3 -c 'print(1)'", work)
        assert control.returncode == 0, control.stderr
        assert _docker(cid, "cat facts.json", work).stdout == "pwned-user\n"
        restored = _docker(cid, "printf 'old-facts\\n' > facts.json", work)
        assert restored.returncode == 0, restored.stderr
        legal = (
            f"python3 {writer} --kind run-card --run-dir {work} "
            f"--relative run-card.json --input {work}/in.json"
        )
        wrote = _docker(
            cid,
            _session(guard_command(legal, env_type="docker", home=str(home)), Path(work)),
            work,
        )
        assert wrote.returncode == 0, wrote.stderr
        assert json.loads(_docker(cid, "cat run-card.json", work).stdout)["marker"] == "legal"
        assert _docker(cid, "cat facts.json", work).stdout == "old-facts\n"
    finally:
        extra = outside if "outside" in locals() else ""
        leftovers = " ".join(
            shlex.quote(path)
            for path in (
                work,
                "/tmp/fake-py",
                extra,
                locals().get("fd_link", ""),
                locals().get("other_dir", ""),
            )
            if path
        )
        _docker(cid, "rm -rf " + leftovers, "/tmp")
        if "site_dir" in locals() and site_dir.startswith("/"):
            subprocess.run(
                ["docker", "exec", "-u", "0", cid, "rm", "-f", site_dir + "/usercustomize.py"],
                capture_output=True, text=True, timeout=30,
            )
