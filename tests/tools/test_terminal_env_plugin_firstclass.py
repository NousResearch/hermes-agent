"""Tests for plugin-registered terminal backends being first-class in core.

Covers the four classification sites that historically used literal frozensets
of built-in backend names and now consult the terminal environment registry so a
plugin-registered backend (``is_container=True``, ``skip_container_guards``,
``cache_path_base``) is treated the same as a built-in one:

1. ``tools.file_tools`` per-task cwd override guard
2. ``tools.approval._should_skip_container_guards``
3. ``tools.terminal_tool._container_config_from_config`` key pass-through
4. ``tools.credential_files.from_agent_visible_cache_path`` reverse translation
"""

import pytest

from agent.terminal_env_provider import TerminalEnvironmentProvider
from agent import terminal_env_registry as reg


class _Env:
    def execute(self, command, **kwargs):
        return {"output": "", "exit_code": 0}

    def cleanup(self):
        pass


class _Provider(TerminalEnvironmentProvider):
    name = "testbox"
    is_remote = True
    is_container = True

    @property
    def cache_path_base(self):
        return "~/.hermes"

    def is_available(self):
        return True

    def create_environment(self, *, cwd, timeout, task_id="default",
                           image=None, container_config=None, **kwargs):
        return _Env()


@pytest.fixture(autouse=True)
def _clean_registry():
    reg._reset_for_tests()
    yield
    reg._reset_for_tests()


def _register(**overrides):
    class P(_Provider):
        pass
    for k, v in overrides.items():
        setattr(P, k, v)
    reg.register_provider(P())
    return P


class TestContainerBackendClassification:
    """The predicate the file-tool cwd guard now relies on is registry-aware."""

    def test_is_container_backend_true_for_plugin_backend(self):
        import tools.terminal_tool as tt

        _register()
        assert tt._is_container_backend("testbox") is True

    def test_is_container_backend_false_when_provider_says_no(self):
        import tools.terminal_tool as tt

        _register(is_container=False)
        assert tt._is_container_backend("testbox") is False


class TestSkipContainerGuards:
    """approval._should_skip_container_guards consults the provider flag."""

    def test_plugin_backend_skips_when_flag_true(self):
        import tools.approval as A

        _register()  # skip_container_guards defaults to is_container == True
        assert A._should_skip_container_guards("testbox") is True

    def test_plugin_backend_does_not_skip_when_flag_false(self):
        import tools.approval as A

        _register(skip_container_guards=False)
        assert A._should_skip_container_guards("testbox") is False

    def test_unknown_backend_never_skips(self):
        import tools.approval as A

        assert A._should_skip_container_guards("no_such_backend") is False


class TestContainerConfigPassthrough:
    """_container_config_from_config carries unknown (plugin) keys through."""

    def test_plugin_keys_ride_container_config(self):
        import tools.terminal_tool as tt

        config = {
            "container_cpu": 2,
            "testbox_api_key": "secret",
            "testbox_region": "eu-west",
        }
        cc = tt._container_config_from_config(config)
        assert cc["testbox_api_key"] == "secret"
        assert cc["testbox_region"] == "eu-west"

    def test_known_keys_still_defaulted(self):
        import tools.terminal_tool as tt

        cc = tt._container_config_from_config({})
        assert cc["container_cpu"] == 1
        assert cc["docker_volumes"] == []


class TestReverseCachePathTranslation:
    """from_agent_visible_cache_path consults the provider's cache_path_base."""

    def test_plugin_backend_reverse_translates(self, monkeypatch):
        from tools import credential_files as cf

        monkeypatch.setenv("TERMINAL_ENV", "testbox")
        _register(cache_path_base="~/.hermes")

        monkeypatch.setattr(
            cf,
            "get_cache_directory_mounts",
            lambda container_base="/root/.hermes": [
                {
                    "host_path": "/host/cache/images",
                    "container_path": f"{container_base}/cache/images",
                }
            ],
        )

        # A container path under the plugin's cache base reverse-maps to host.
        assert cf.from_agent_visible_cache_path(
            "~/.hermes/cache/images/file.png"
        ) == "/host/cache/images/file.png"
        # A path outside any mount is returned unchanged.
        assert cf.from_agent_visible_cache_path(
            "~/.hermes/cache/other/file.png"
        ) == "~/.hermes/cache/other/file.png"

    def test_plugin_backend_without_cache_base_returns_unchanged(self, monkeypatch):
        from tools import credential_files as cf

        monkeypatch.setenv("TERMINAL_ENV", "testbox")
        _register(cache_path_base=None)

        path = "/root/.hermes/cache/images/file.png"
        assert cf.from_agent_visible_cache_path(path) == path

    def test_unknown_backend_returns_unchanged(self, monkeypatch):
        from tools import credential_files as cf

        monkeypatch.setenv("TERMINAL_ENV", "no_such_backend")
        path = "/root/.hermes/cache/images/file.png"
        assert cf.from_agent_visible_cache_path(path) == path
