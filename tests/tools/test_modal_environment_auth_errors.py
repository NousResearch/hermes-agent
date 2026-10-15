"""Diagnosis of Modal authentication failures (#47264).

On Termux the reporter's Modal tool calls died with a bare
``modal.exception.AuthError: Token ID is malformed`` and nothing said which
credential source the SDK actually used: Hermes publishes ``$HERMES_HOME/.env``
into ``os.environ``, and the Modal SDK prefers
MODAL_TOKEN_ID/MODAL_TOKEN_SECRET from the environment over the config file at
MODAL_CONFIG_PATH or ``~/.modal.toml`` that ``modal token info`` (a plain shell,
which never reads Hermes's .env) reports as healthy — per variable, decided by
membership in ``os.environ`` (an empty ``MODAL_TOKEN_ID=""`` still counts as
the env credential). A broken .env pair therefore silently beats a working
profile token, and with no credentials at all the SDK's 219-char "Token
missing…" novel must not push the diagnosis toward the tool layer's real
bound: ``tools/registry.py::_MAX_TOOL_ERROR_CHARS = 2048`` with a hard
mid-word cut (after ``terminal_tool`` prefixes "Failed to execute command: ").

The contract under test: a Modal auth failure surfaces as a RuntimeError whose
first line is the server message and whose body names the credential source —
the env pair, or the config file at the resolved path described only as where
the SDK looked, never as what it acquired (a partial pair with no usable file
gets "no complete credential pair", never a phantom source), or plainly "no
credentials found" when neither exists — keeps the SDK exception
reachable as ``__cause__``, stays within 640 chars in the worst case (detail
capped, 60-char path) so the Fix line survives long before the 2048 hard cut,
caps the parenthetical detail on a word boundary, and — because auth dies
before any sandbox exists — leaves the recorded snapshot id intact for the next
session while stopping the worker. The parenthetical detail caps at a word
boundary only when the trim keeps at least half the cap — otherwise it takes a
hard cut, so a long unbroken token is never thrown away wholesale.
"""

import json
import os
import sys
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from tools.environments import modal as modal_env

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"

SERVER_MESSAGE = "Token ID is malformed"
# Verbatim SDK "Token missing" text (client.py), measured at 219 chars — the
# cap must word-boundary-truncate this novel, not reproduce it whole.
LONG_SDK_MESSAGE = (
    "Token missing. Could not authenticate client. If you have token "
    "credentials, see modal.com/docs/reference/modal.config for setup help. "
    "If you are a new user, register an account at modal.com, then run "
    "`modal token new`.")
assert len(LONG_SDK_MESSAGE) == 219


class FakeAuthError(Exception):
    """Stand-in for modal.exception.AuthError — the real SDK is never imported."""


def _load_module(module_name: str, path: Path):
    spec = spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _reset_modules(prefixes: tuple[str, ...]):
    for name in list(sys.modules):
        if name.startswith(prefixes):
            sys.modules.pop(name, None)


@pytest.fixture(autouse=True)
def _restore_tool_modules():
    original_hermes_home = os.environ.get("HERMES_HOME")
    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "tools"
        or name.startswith("tools.")
        or name == "hermes_cli"
        or name.startswith("hermes_cli.")
        or name == "modal"
        or name.startswith("modal.")
    }
    try:
        yield
    finally:
        if original_hermes_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = original_hermes_home
        _reset_modules(("tools", "hermes_cli", "modal"))
        sys.modules.update(original_modules)


def _install_fake_sdk(tmp_path: Path):
    """Fake the Modal SDK and Hermes plumbing (mirrors the harness in
    test_modal_snapshot_isolation.py). The fake modal namespace exposes
    ``exception.AuthError`` because the constructor's except clause evaluates
    that attribute, and ``App.lookup`` raises it — auth fails at the first
    RPC, so this world has no Sandbox.create at all."""
    _reset_modules(("tools", "hermes_cli", "modal"))

    hermes_cli = types.ModuleType("hermes_cli")
    hermes_cli.__path__ = [str(REPO_ROOT / "hermes_cli")]  # type: ignore[attr-defined]
    sys.modules["hermes_cli"] = hermes_cli
    hermes_home = tmp_path / "hermes-home"
    os.environ["HERMES_HOME"] = str(hermes_home)
    sys.modules["hermes_cli.config"] = types.SimpleNamespace(
        get_hermes_home=lambda: hermes_home,
    )

    tools_package = types.ModuleType("tools")
    tools_package.__path__ = [str(TOOLS_DIR)]  # type: ignore[attr-defined]
    sys.modules["tools"] = tools_package

    env_package = types.ModuleType("tools.environments")
    env_package.__path__ = [str(TOOLS_DIR / "environments")]  # type: ignore[attr-defined]
    sys.modules["tools.environments"] = env_package

    # The faked modal module answers every SDK touch; the real lazy-dep gate
    # (a version-pinned metadata check) must not refuse first without the
    # modal extra installed.
    sys.modules["tools.lazy_deps"] = types.SimpleNamespace(ensure=lambda *a, **k: None)

    class _DummyBaseEnvironment:
        def __init__(self, cwd: str, timeout: int, env=None):
            self.cwd = cwd
            self.timeout = timeout
            self.env = env or {}

        def _prepare_command(self, command: str):
            return command, None

        def init_session(self):
            pass

    class _DummyThreadedProcessHandle:
        def __init__(self, exec_fn, cancel_fn=None):
            pass

    def _load_json_store(path):
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                pass
        return {}

    def _save_json_store(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    def _file_mtime_key(host_path):
        try:
            st = Path(host_path).stat()
            return (st.st_mtime, st.st_size)
        except OSError:
            return None

    sys.modules["tools.environments.base"] = types.SimpleNamespace(
        BaseEnvironment=_DummyBaseEnvironment,
        _ThreadedProcessHandle=_DummyThreadedProcessHandle,
        _load_json_store=_load_json_store,
        _save_json_store=_save_json_store,
        _file_mtime_key=_file_mtime_key,
    )
    sys.modules["tools.interrupt"] = types.SimpleNamespace(is_interrupted=lambda: False)
    sys.modules["tools.credential_files"] = types.SimpleNamespace(
        get_credential_file_mounts=lambda: [],
        iter_skills_files=lambda **kw: [],
        iter_cache_files=lambda **kw: [],
    )

    lookup_calls: list[str] = []

    class _FakeImage:
        @staticmethod
        def from_id(image_id: str):
            return {"kind": "snapshot", "image_id": image_id}

        @staticmethod
        def from_registry(image: str, setup_dockerfile_commands=None):
            return {"kind": "registry", "image": image}

    async def _lookup_aio(name: str, create_if_missing: bool = False):
        # AppGetOrCreate is the first RPC the SDK makes; auth dies right here.
        lookup_calls.append(name)
        raise FakeAuthError(SERVER_MESSAGE)

    class _FakeApp:
        lookup = types.SimpleNamespace(aio=_lookup_aio)

    sys.modules["modal"] = types.SimpleNamespace(
        Image=_FakeImage,
        App=_FakeApp,
        exception=types.SimpleNamespace(AuthError=FakeAuthError),
    )

    return {
        "hermes_home": hermes_home,
        "snapshot_store": hermes_home / "modal_snapshots.json",
        "lookup_calls": lookup_calls,
    }


# --- the pure diagnosis function (no SDK needed) --------------------------------

def test_auth_diagnosis_names_the_credential_source_the_sdk_actually_read(
        tmp_path, monkeypatch):
    # environ/config_path are injected so nothing here touches real env or
    # files — each branch sees only the world the SDK would have seen.
    monkeypatch.delenv("HERMES_HOME", raising=False)
    present = tmp_path / "present-modal.toml"
    present.write_text("[default]\ntoken_id = 'tn-unknown'\n")
    absent = tmp_path / "absent-modal.toml"  # deliberately never created

    # Branch A on a PARTIAL pair must name the variable actually present as
    # the env source and say where the SDK LOOKED for the missing half —
    # never that the half "comes from" the file: an existing file may still
    # hold nothing usable (profile section absent, no token entry), and a
    # diagnostic must not parse the user's TOML to tell the difference.
    # The resolved path is named, never a hardcoded ~/.modal.toml.
    branch_a = modal_env._diagnose_modal_auth_error(
        ValueError(SERVER_MESSAGE),
        environ={"MODAL_TOKEN_ID": "tk-secret-looking-value"},
        config_path=str(present))
    assert SERVER_MESSAGE in branch_a and "MODAL_TOKEN_ID" in branch_a
    # the resolved path, never a hardcoded one: a user with MODAL_CONFIG_PATH
    # set must not be told about a file the SDK never read
    assert str(present) in branch_a and "~/.modal.toml" not in branch_a
    assert "tk-secret-looking-value" not in branch_a  # names/paths only, redaction-safe
    source_a = branch_a.splitlines()[1]
    assert "MODAL_TOKEN_ID in the environment" in source_a
    assert ("MODAL_TOKEN_SECRET is not set, so the SDK looks for that half in"
            in source_a)
    assert "comes from" not in source_a  # where it looked, not what it acquired
    assert str(present) in source_a
    assert len(branch_a) <= 640

    # When the config file does NOT exist, the SDK opens nothing for the
    # missing half, so from_env builds no credentials and raises "Token
    # missing…". "supplied nothing" states what the SDK found — truthful for
    # an absent, unreadable or unexpanded-~ path alike — and never names the
    # file as the source of the half.
    branch_a_no_file = modal_env._diagnose_modal_auth_error(
        ValueError(SERVER_MESSAGE),
        environ={"MODAL_TOKEN_ID": "tk-secret-looking-value"},
        config_path=str(absent))
    source_a_no_file = branch_a_no_file.splitlines()[1]
    assert "MODAL_TOKEN_ID in the environment" in source_a_no_file
    assert "comes from" not in source_a_no_file  # no phantom file source
    assert "MODAL_TOKEN_SECRET is not set and" in source_a_no_file
    assert f"{absent} supplied nothing" in source_a_no_file
    assert "no complete credential pair" in source_a_no_file
    assert "`modal token info`" in source_a_no_file
    assert len(branch_a_no_file) <= 640

    # Branch A on a COMPLETE pair must tell the truth too: with both halves
    # in the environment the SDK never reads the file, so no wording may
    # imply the file still contributes.
    branch_a_complete = modal_env._diagnose_modal_auth_error(
        ValueError(SERVER_MESSAGE),
        environ={"MODAL_TOKEN_ID": "tk-a", "MODAL_TOKEN_SECRET": "ts-b"},
        config_path=str(present))
    source_a_complete = branch_a_complete.splitlines()[1]
    assert ("MODAL_TOKEN_ID/MODAL_TOKEN_SECRET in the environment supplies "
            "both halves") in source_a_complete
    assert f"nothing is read from {present}" in source_a_complete
    assert "is not set" not in source_a_complete

    branch_b = modal_env._diagnose_modal_auth_error(
        ValueError("Token not found"), environ={}, config_path=str(present))
    assert "Token not found" in branch_b
    assert f"the Modal profile in {present}" in branch_b
    assert "modal token new" in branch_b

    branch_c = modal_env._diagnose_modal_auth_error(
        ValueError(LONG_SDK_MESSAGE), environ={}, config_path=str(absent))
    # no file exists AND no env var: the diagnosis must not present a
    # profile/file as "the source" — the only config-path mention is the
    # "no config file at" clause, and the SDK novel is capped, not reproduced
    assert "no Modal credentials found" in branch_c
    assert str(absent) in branch_c
    assert "Source: no Modal credentials found" in branch_c
    assert LONG_SDK_MESSAGE not in branch_c

    # The SDK decides by MEMBERSHIP (`env_var_key in os.environ`,
    # modal/config.py Config.get), so an empty MODAL_TOKEN_ID="" IS the env
    # credential it read — branch A naming the environment, not branch C.
    branch_empty = modal_env._diagnose_modal_auth_error(
        ValueError(SERVER_MESSAGE), environ={"MODAL_TOKEN_ID": ""},
        config_path=str(absent))
    assert "MODAL_TOKEN_ID in the environment" in branch_empty
    assert "no Modal credentials found" not in branch_empty

    # Default path resolution (no config_path injected) is part of the
    # contract: MODAL_CONFIG_PATH honoured, else expanduser("~/.modal.toml")
    # — the exact modal/config.py rule. Without these, breaking the default
    # while every other test passes config_path= explicitly would stay green.
    branch_modalcfg = modal_env._diagnose_modal_auth_error(
        ValueError(SERVER_MESSAGE), environ={"MODAL_CONFIG_PATH": str(present)})
    assert f"the Modal profile in {present}" in branch_modalcfg

    branch_default = modal_env._diagnose_modal_auth_error(
        ValueError(SERVER_MESSAGE), environ={})
    expanded_default = os.path.expanduser("~/.modal.toml")
    assert expanded_default in branch_default
    assert "~/.modal.toml" not in branch_default  # expanded, never literal


def test_auth_diagnosis_worst_case_fits_budget_and_caps_detail_on_words(monkeypatch):
    # Worst case: both vars present, a raw 500-char detail that must cap, and
    # a 60-char resolved config path. Pin HERMES_HOME so the displayed .env
    # path is deterministic everywhere.
    monkeypatch.setenv("HERMES_HOME", str(Path.home() / ".hermes/profiles/coder"))
    # word layout guarantees a naive 200-char cut would chop a word mid-way
    raw_detail = (" ".join(["abcdef"] * 72))[:500]
    assert len(raw_detail) == 500
    path60 = "/m/" + "d" * 55 + "/a"
    assert len(path60) == 60
    message = modal_env._diagnose_modal_auth_error(
        ValueError(raw_detail),
        environ={"MODAL_TOKEN_ID": "tk-a", "MODAL_TOKEN_SECRET": "ts-b"},
        config_path=path60)
    # Far inside the tool layer's real cap (registry._MAX_TOOL_ERROR_CHARS =
    # 2048, a hard mid-word cut) so the Fix line always survives intact.
    assert len(message) <= 640
    fix_line = message.splitlines()[-1]
    assert fix_line.startswith("Fix:") and "MODAL_TOKEN_ID" in fix_line

    # The partial-pair sub-cases carry longer tails; the no-file one is the
    # longest wording of all, so it must fit the SAME pinned worst case.
    no_file = modal_env._diagnose_modal_auth_error(
        ValueError(raw_detail), environ={"MODAL_TOKEN_ID": "tk-a"},
        config_path=path60)
    assert len(no_file) <= 640 and f"{path60} supplied nothing" in no_file

    # Check the parenthetical DETAIL span itself: it ends with the ellipsis,
    # the kept span is a verbatim prefix of the raw text, and the character
    # the raw text has at the cut index is whitespace — the cut lands on a
    # word boundary, so no broken tail.
    first = message.splitlines()[0]
    prefix = "Modal authentication failed ("
    assert first.startswith(prefix) and first.endswith(").")
    detail = first[len(prefix):-2]
    assert detail.endswith("…")
    kept = detail[:-1]
    assert raw_detail.startswith(kept)
    assert raw_detail[len(kept)] == " "  # cut landed on whitespace: no broken tail

    # Truncation shape 1 — space near the START: "Error: " plus a 200-char
    # unbroken token. A naive last-word trim would keep only "Error:…" and
    # throw the diagnosis away; the half-limit rule must hard-cut instead,
    # retaining the bulk of the specific token.
    token = "X" * 200
    spacey_detail = modal_env._diagnose_modal_auth_error(
        ValueError("Error: " + token), environ={}, config_path="/nonexistent/x.toml"
    ).splitlines()[0]
    kept_spacey = spacey_detail[len(prefix):-3]
    assert kept_spacey.startswith("Error: ") and len(kept_spacey) == 200
    assert token[:150] in kept_spacey  # the specific content survives

    # Truncation shape 2 — NO space at all in the slice: rsplit finds nothing,
    # the result is a hard 200-char cut plus the ellipsis.
    token2 = "Y" * 250
    spaceless_detail = modal_env._diagnose_modal_auth_error(
        ValueError(token2), environ={}, config_path="/nonexistent/x.toml"
    ).splitlines()[0]
    kept_spaceless = spaceless_detail[len(prefix):-3]
    assert kept_spaceless == token2[:200]  # hard cut, no word lost
    assert spaceless_detail.endswith("Y…).")


# --- the constructor path, through the fake SDK ---------------------------------

def test_auth_failure_becomes_diagnosed_runtime_error_preserving_cause(tmp_path, monkeypatch):
    monkeypatch.setenv("MODAL_TOKEN_ID", "tk-not-an-id")
    state = _install_fake_sdk(tmp_path)

    modal_module = _load_module("tools.environments.modal", TOOLS_DIR / "environments" / "modal.py")

    class _RecordingWorker(modal_module._AsyncWorker):
        """Counts stop() calls: the auth arm runs on a daemon thread, so only
        an explicit counter proves the worker is actually stopped."""
        stops = 0

        def stop(self):
            type(self).stops += 1
            super().stop()

    monkeypatch.setattr(modal_module, "_AsyncWorker", _RecordingWorker)
    with pytest.raises(RuntimeError) as excinfo:
        modal_module.ModalEnvironment(image="python:3.11", task_id="auth-task")

    assert state["lookup_calls"] == ["hermes-agent"]  # we really reached the SDK call
    # the diagnosis, not the bare server string: env source is named
    assert str(excinfo.value).startswith(f"Modal authentication failed ({SERVER_MESSAGE}).")
    assert "MODAL_TOKEN_ID" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, FakeAuthError)  # chain preserved
    # exactly once: the auth arm stops the worker, and the inner retry arm's
    # generic handler must not double-stop it
    assert _RecordingWorker.stops == 1


def test_auth_failure_on_snapshot_path_keeps_the_recorded_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("MODAL_TOKEN_ID", "tk-not-an-id")
    state = _install_fake_sdk(tmp_path)
    state["snapshot_store"].parent.mkdir(parents=True, exist_ok=True)
    state["snapshot_store"].write_text(json.dumps({"direct:snap-task": "im-keepme"}))

    modal_module = _load_module("tools.environments.modal", TOOLS_DIR / "environments" / "modal.py")
    with pytest.raises(RuntimeError):
        modal_module.ModalEnvironment(image="python:3.11", task_id="snap-task")

    # auth fails before any sandbox exists, so the snapshot was never suspect
    assert json.loads(state["snapshot_store"].read_text()) == {"direct:snap-task": "im-keepme"}
