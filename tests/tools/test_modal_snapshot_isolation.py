import json
import sys
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


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


def _install_modal_test_modules(
    tmp_path: Path,
    *,
    fail_on_snapshot_ids: set[str] | None = None,
    snapshot_id: str = "im-fresh",
):
    _reset_modules(("tools", "hermes_cli", "swerex", "modal"))

    hermes_cli = types.ModuleType("hermes_cli")
    hermes_cli.__path__ = []  # type: ignore[attr-defined]
    sys.modules["hermes_cli"] = hermes_cli
    hermes_home = tmp_path / "hermes-home"
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

    sys.modules["tools.environments.base"] = types.SimpleNamespace(BaseEnvironment=_DummyBaseEnvironment)
    sys.modules["tools.interrupt"] = types.SimpleNamespace(is_interrupted=lambda: False)

    from_id_calls: list[str] = []
    registry_calls: list[tuple[str, list[str] | None]] = []
    deployment_calls: list[dict] = []

    class _FakeImage:
        @staticmethod
        def from_id(image_id: str):
            from_id_calls.append(image_id)
            return {"kind": "snapshot", "image_id": image_id}

        @staticmethod
        def from_registry(image: str, setup_dockerfile_commands=None):
            registry_calls.append((image, setup_dockerfile_commands))
            return {"kind": "registry", "image": image}

    class _FakeRuntime:
        async def execute(self, _command):
            return types.SimpleNamespace(stdout="ok", exit_code=0)

    class _FakeModalDeployment:
        def __init__(self, **kwargs):
            deployment_calls.append(dict(kwargs))
            self.image = kwargs["image"]
            self.runtime = _FakeRuntime()

            async def _snapshot_aio():
                return types.SimpleNamespace(object_id=snapshot_id)

            self._sandbox = types.SimpleNamespace(
                snapshot_filesystem=types.SimpleNamespace(aio=_snapshot_aio),
            )

        async def start(self):
            image = self.image if isinstance(self.image, dict) else {}
            image_id = image.get("image_id")
            if fail_on_snapshot_ids and image_id in fail_on_snapshot_ids:
                raise RuntimeError(f"cannot restore {image_id}")

        async def stop(self):
            return None

    class _FakeRexCommand:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    sys.modules["modal"] = types.SimpleNamespace(Image=_FakeImage)

    swerex = types.ModuleType("swerex")
    swerex.__path__ = []  # type: ignore[attr-defined]
    sys.modules["swerex"] = swerex
    swerex_deployment = types.ModuleType("swerex.deployment")
    swerex_deployment.__path__ = []  # type: ignore[attr-defined]
    sys.modules["swerex.deployment"] = swerex_deployment
    sys.modules["swerex.deployment.modal"] = types.SimpleNamespace(ModalDeployment=_FakeModalDeployment)
    swerex_runtime = types.ModuleType("swerex.runtime")
    swerex_runtime.__path__ = []  # type: ignore[attr-defined]
    sys.modules["swerex.runtime"] = swerex_runtime
    sys.modules["swerex.runtime.abstract"] = types.SimpleNamespace(Command=_FakeRexCommand)

    return {
        "snapshot_store": hermes_home / "modal_snapshots.json",
        "deployment_calls": deployment_calls,
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
        assert state["deployment_calls"][0]["image"] == {"kind": "snapshot", "image_id": "im-legacy123"}
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
        assert [call["image"] for call in state["deployment_calls"]] == [
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
