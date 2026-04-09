import json
import os
import sys
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"


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


def _install_modal_test_modules(
    tmp_path: Path,
    *,
    fail_on_snapshot_ids: set[str] | None = None,
    snapshot_id: str = "im-fresh",
):
    _reset_modules(("tools", "hermes_cli", "modal"))

    hermes_cli = types.ModuleType("hermes_cli")
    hermes_cli.__path__ = []  # type: ignore[attr-defined]
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

    class _DummyBaseEnvironment:
        def __init__(self, cwd: str, timeout: int, env=None):
            self.cwd = cwd
            self.timeout = timeout
            self.env = env or {}

        def _prepare_command(self, command: str):
            return command, None

        def init_session(self):
            pass

    # Stub _ThreadedProcessHandle: modal.py imports it but only uses it at
    # runtime inside _run_bash; the snapshot-isolation tests never call _run_bash,
    # so a class placeholder is sufficient.
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

    from_id_calls: list[str] = []
    registry_calls: list[tuple[str, list[str] | None]] = []
    create_calls: list[dict] = []

    class _FakeImage:
        @staticmethod
        def from_id(image_id: str):
            from_id_calls.append(image_id)
            return {"kind": "snapshot", "image_id": image_id}

        @staticmethod
        def from_registry(image: str, setup_dockerfile_commands=None):
            registry_calls.append((image, setup_dockerfile_commands))
            return {"kind": "registry", "image": image}

    async def _lookup_aio(_name: str, create_if_missing: bool = False):
        return types.SimpleNamespace(name="hermes-agent", create_if_missing=create_if_missing)

    class _FakeSandboxInstance:
        def __init__(self, image):
            self.image = image

            async def _snapshot_aio():
                return types.SimpleNamespace(object_id=snapshot_id)

            async def _terminate_aio():
                return None

            self.snapshot_filesystem = types.SimpleNamespace(aio=_snapshot_aio)
            self.terminate = types.SimpleNamespace(aio=_terminate_aio)

    async def _create_aio(*_args, image=None, app=None, timeout=None, **kwargs):
        create_calls.append({
            "image": image,
            "app": app,
            "timeout": timeout,
            **kwargs,
        })
        image_id = image.get("image_id") if isinstance(image, dict) else None
        if fail_on_snapshot_ids and image_id in fail_on_snapshot_ids:
            raise RuntimeError(f"cannot restore {image_id}")
        return _FakeSandboxInstance(image)

    class _FakeMount:
        @staticmethod
        def from_local_file(host_path: str, remote_path: str):
            return {"host_path": host_path, "remote_path": remote_path}

    class _FakeApp:
        lookup = types.SimpleNamespace(aio=_lookup_aio)

    class _FakeSandbox:
        create = types.SimpleNamespace(aio=_create_aio)

    sys.modules["modal"] = types.SimpleNamespace(
        Image=_FakeImage,
        App=_FakeApp,
        Sandbox=_FakeSandbox,
        Mount=_FakeMount,
    )

    return {
        "snapshot_store": hermes_home / "modal_snapshots.json",
        "create_calls": create_calls,
        "from_id_calls": from_id_calls,
        "registry_calls": registry_calls,
    }


def test_modal_environment_migrates_legacy_snapshot_key_and_uses_snapshot_id(tmp_path):
    state = _install_modal_test_modules(tmp_path)
    snapshot_store = state["snapshot_store"]
    snapshot_store.parent.mkdir(parents=True, exist_ok=True)
    snapshot_store.write_text(json.dumps({"task-legacy": "im-legacy123"}))

    modal_module = _load_module("tools.environments.modal", TOOLS_DIR / "environments" / "modal.py")
    env = modal_module.ModalEnvironment(image="python:3.11", task_id="task-legacy")

    try:
        assert state["from_id_calls"] == ["im-legacy123"]
        assert state["create_calls"][0]["image"] == {"kind": "snapshot", "image_id": "im-legacy123"}
        assert json.loads(snapshot_store.read_text()) == {"direct:task-legacy": "im-legacy123"}
    finally:
        env.cleanup()


def test_modal_environment_prunes_stale_direct_snapshot_and_retries_base_image(tmp_path):
    state = _install_modal_test_modules(tmp_path, fail_on_snapshot_ids={"im-stale123"})
    snapshot_store = state["snapshot_store"]
    snapshot_store.parent.mkdir(parents=True, exist_ok=True)
    snapshot_store.write_text(json.dumps({"direct:task-stale": "im-stale123"}))

    modal_module = _load_module("tools.environments.modal", TOOLS_DIR / "environments" / "modal.py")
    env = modal_module.ModalEnvironment(image="python:3.11", task_id="task-stale")

    try:
        assert [call["image"] for call in state["create_calls"]] == [
            {"kind": "snapshot", "image_id": "im-stale123"},
            {"kind": "registry", "image": "python:3.11"},
        ]
        assert json.loads(snapshot_store.read_text()) == {}
    finally:
        env.cleanup()


def test_modal_environment_cleanup_writes_namespaced_snapshot_key(tmp_path):
    state = _install_modal_test_modules(tmp_path, snapshot_id="im-cleanup456")
    snapshot_store = state["snapshot_store"]

    modal_module = _load_module("tools.environments.modal", TOOLS_DIR / "environments" / "modal.py")
    env = modal_module.ModalEnvironment(image="python:3.11", task_id="task-cleanup")
    env.cleanup()

    assert json.loads(snapshot_store.read_text()) == {"direct:task-cleanup": "im-cleanup456"}


def test_resolve_modal_image_uses_snapshot_ids_and_registry_images(tmp_path):
    state = _install_modal_test_modules(tmp_path)
    modal_module = _load_module("tools.environments.modal", TOOLS_DIR / "environments" / "modal.py")

    snapshot_image = modal_module._resolve_modal_image("im-snapshot123")
    registry_image = modal_module._resolve_modal_image("python:3.11")

    assert snapshot_image == {"kind": "snapshot", "image_id": "im-snapshot123"}
    assert registry_image == {"kind": "registry", "image": "python:3.11"}
    assert state["from_id_calls"] == ["im-snapshot123"]
    assert state["registry_calls"][0][0] == "python:3.11"
    assert "ensurepip" in state["registry_calls"][0][1][0]
