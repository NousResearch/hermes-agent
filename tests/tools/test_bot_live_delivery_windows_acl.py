"""#139485 / #138872 on real Windows ACLs.

The reporters' mailbox dir was created by an elevated process: owner BUILTIN\\Administrators, a DACL
with no entry for the user. Their non-elevated gateway can't chmod it, list it or open ``.lock`` in
it, and startup must skip Bot Chat recovery instead of aborting. A second, milder shape (only
FILE_WRITE_ATTRIBUTES denied) leaves the dir usable, and the lock must keep working there.

CPython opens dirs with FILE_FLAG_BACKUP_SEMANTICS, so a token with SeBackupPrivilege/SeRestorePrivilege
enabled (an elevated CI runner) bypasses every ACE. The reporter's token has neither. Each probe runs in a
child that removes both privileges from its own token first, then exercises the production code.
"""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("windows")

_REPO = Path(__file__).resolve().parents[2]

_DROP_PRIVILEGES = textwrap.dedent(r"""
    import ctypes, ctypes.wintypes as w, json, subprocess, sys
    from pathlib import Path

    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)

    class LUID(ctypes.Structure):
        _fields_ = [("LowPart", w.DWORD), ("HighPart", w.LONG)]

    class LUID_AND_ATTRIBUTES(ctypes.Structure):
        _fields_ = [("Luid", LUID), ("Attributes", w.DWORD)]

    class TOKEN_PRIVILEGES(ctypes.Structure):
        _fields_ = [("PrivilegeCount", w.DWORD), ("Privileges", LUID_AND_ATTRIBUTES * 1)]

    kernel.GetCurrentProcess.restype = w.HANDLE
    advapi.OpenProcessToken.argtypes = [w.HANDLE, w.DWORD, ctypes.POINTER(w.HANDLE)]
    advapi.LookupPrivilegeValueW.argtypes = [w.LPCWSTR, w.LPCWSTR, ctypes.POINTER(LUID)]
    advapi.AdjustTokenPrivileges.argtypes = [w.HANDLE, w.BOOL, ctypes.POINTER(TOKEN_PRIVILEGES),
                                             w.DWORD, ctypes.c_void_p, ctypes.c_void_p]
    token = w.HANDLE()
    if not advapi.OpenProcessToken(kernel.GetCurrentProcess(), 0x0020 | 0x0008, ctypes.byref(token)):
        raise ctypes.WinError(ctypes.get_last_error())
    for name in ("SeBackupPrivilege", "SeRestorePrivilege"):
        luid = LUID()
        if not advapi.LookupPrivilegeValueW(None, name, ctypes.byref(luid)):
            raise ctypes.WinError(ctypes.get_last_error())
        # SE_PRIVILEGE_REMOVED: gone from this process's token for good, as in a non-elevated token.
        tp = TOKEN_PRIVILEGES(1, (LUID_AND_ATTRIBUTES * 1)(LUID_AND_ATTRIBUTES(luid, 0x00000004)))
        if not advapi.AdjustTokenPrivileges(token, False, ctypes.byref(tp), 0, None, None):
            raise ctypes.WinError(ctypes.get_last_error())
""")

_CHILD = _DROP_PRIVILEGES + textwrap.dedent(r"""
    from tools import bot_live_delivery as mailbox

    home = Path(sys.argv[1])
    root = home / "runtime" / mailbox.DELIVERY_DIR_NAME
    try:
        root.chmod(0o700)
        denied = False
    except PermissionError:
        denied = True
    if not denied:
        priv = subprocess.run(["whoami", "/priv"], capture_output=True, text=True, check=False).stdout
        print(json.dumps({"denied": False, "priv": priv}))
        sys.exit(0)
    with mailbox._locked(home) as locked:
        mailbox._write(locked / ("a" * 32 + ".json"), {"status": "queued"})
    print(json.dumps({"denied": True, "read": mailbox._read(root / ("a" * 32 + ".json")),
                      "locked": str(locked)}))
""")


def test_mailbox_lock_survives_a_dir_whose_acl_denies_write_attributes(tmp_path):
    root = tmp_path / "runtime" / "bot_live_delivery"
    root.mkdir(parents=True)
    # Deny write-attributes to Everyone (S-1-1-0): no group or name form of the token escapes it.
    subprocess.run(["icacls", str(root), "/deny", "*S-1-1-0:(WA)"], check=True, capture_output=True)
    try:
        child = subprocess.run([sys.executable, "-c", _CHILD, str(tmp_path)], cwd=_REPO,
                               capture_output=True, text=True, check=False, timeout=120)
        assert child.returncode == 0, f"the production lock failed under the reporter's ACL:\n{child.stderr}"
        result = json.loads(child.stdout.strip().splitlines()[-1])
        assert result["denied"], f"chmod still succeeded without backup/restore privileges:\n{result['priv']}"
        assert Path(result["locked"]) == root.resolve()
        assert result["read"] == {"status": "queued"}
    finally:
        subprocess.run(["icacls", str(root), "/remove:d", "*S-1-1-0"], check=False, capture_output=True)


_UNUSABLE_CHILD = _DROP_PRIVILEGES + textwrap.dedent(r"""
    import asyncio, logging, os
    from types import SimpleNamespace

    home = Path(sys.argv[1])
    root = home / "runtime" / "bot_live_delivery"
    try:
        os.close(os.open(root / ".lock", os.O_CREAT | os.O_WRONLY, 0o600))
        denied = False
    except PermissionError:
        denied = True
    if not denied:
        priv = subprocess.run(["whoami", "/user", "/groups", "/priv"], capture_output=True, text=True,
                              check=False).stdout
        acl = subprocess.run(["icacls", str(root)], capture_output=True, text=True, check=False).stdout
        print(json.dumps({"denied": False, "priv": priv + "\n" + acl}))
        sys.exit(0)

    records = []
    handler = logging.Handler()
    handler.emit = lambda record: records.append((record.levelname, record.getMessage()))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.DEBUG)

    from gateway import run_api, run_runtime
    from hermes_state import SessionDB

    async def api(_runner):
        return SimpleNamespace(api_origin="http://127.0.0.1:1")

    run_api.start_gateway_api = api
    db = SessionDB(home / "state.db")
    try:
        runner = SimpleNamespace(session_authority=SimpleNamespace(db=db, waiters={}),
                                 session_runtime_descriptor={})
        # The step start_gateway runs per authority; on main it raised and aborted the gateway.
        asyncio.run(run_runtime.start_gateway_runtime_api(runner))
    finally:
        db.close()
    print(json.dumps({"denied": True, "api_origin": runner.session_runtime_descriptor.get("api_origin"),
                      "logs": [r for r in records if "bot" in r[1].lower()]}))
""")


def test_startup_skips_bot_recovery_on_the_reporters_mailbox_acl(tmp_path):
    root = tmp_path / "runtime" / "bot_live_delivery"
    root.mkdir(parents=True)
    # The reporters' effect: this token can neither list nor create in the dir (#138872: no ACE for the
    # user). Reassigning the owner to an identity outside the runner's token needs SeRestore inside icacls
    # (exit 5 on the hosted runner), and an Administrators-owned, SYSTEM-only DACL still let the runner in.
    # A deny ACE for Everyone is definitive for every identity the runner might hold; with SeBackup and
    # SeRestore removed the child has no bypass, so os.open(.lock) fails as in the reports. The owner's
    # implicit WRITE_DAC is untouched, which is what lets the finally block lift the deny again.
    subprocess.run(["icacls", str(root), "/deny", "*S-1-1-0:(F)"], check=True, capture_output=True)
    try:
        child = subprocess.run([sys.executable, "-c", _UNUSABLE_CHILD, str(tmp_path)], cwd=_REPO,
                               capture_output=True, text=True, check=False, timeout=180)
        assert child.returncode == 0, f"gateway startup aborted on the reporters' ACL:\n{child.stderr}"
        result = json.loads(child.stdout.strip().splitlines()[-1])
        assert result["denied"], f"os.open(.lock) succeeded without backup/restore privileges:\n{result['priv']}"
        assert result["api_origin"] == "http://127.0.0.1:1"
        warnings = [message for level, message in result["logs"]
                    if level == "WARNING" and "not usable by this process" in message]
        assert len(warnings) == 1, result["logs"]
        assert str(root.resolve()) in warnings[0] and "(owner " in warnings[0]
        assert "icacls" in warnings[0] and "/reset" in warnings[0]
    finally:
        subprocess.run(["icacls", str(root), "/remove:d", "*S-1-1-0"], check=False, capture_output=True)
