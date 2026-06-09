from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace


_CONFTEST_PATH = Path(__file__).with_name("conftest.py")
_SPEC = spec_from_file_location("tests_conftest_module", _CONFTEST_PATH)
assert _SPEC and _SPEC.loader
_CONFTEST = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CONFTEST)


class _FakeConfig:
    def __init__(self, timeout_method: str):
        self.option = SimpleNamespace(timeout_method=timeout_method)
        self.markers: list[tuple[str, str]] = []

    def addinivalue_line(self, name: str, value: str) -> None:
        self.markers.append((name, value))


def test_pytest_configure_switches_timeout_method_on_windows(monkeypatch):
    config = _FakeConfig("signal")
    monkeypatch.setattr(_CONFTEST.sys, "platform", "win32", raising=False)

    _CONFTEST.pytest_configure(config)

    assert config.option.timeout_method == "thread"


def test_pytest_configure_keeps_timeout_method_on_non_windows(monkeypatch):
    config = _FakeConfig("signal")
    monkeypatch.setattr(_CONFTEST.sys, "platform", "linux", raising=False)

    _CONFTEST.pytest_configure(config)

    assert config.option.timeout_method == "signal"
