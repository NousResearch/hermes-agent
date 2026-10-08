"""Plugin-registered terminal backends are first-class at two core sites.

1. ``tools.terminal_tool_backends._container_config_from_config`` passes unknown (plugin) keys through.
2. ``tools.credential_files.from_agent_visible_cache_path`` consults the provider's ``cache_path_base``,
   matching its forward twin ``to_agent_visible_cache_path``.

Adapted from @ajzrva-sys's tests on #78035.
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


class TestContainerConfigPassthrough:
    """_container_config_from_config carries unknown (plugin) keys through."""

    def test_plugin_keys_ride_container_config(self):
        import tools.terminal_tool_backends as tt

        config = {
            "container_cpu": 2,
            "testbox_api_key": "secret",
            "testbox_region": "eu-west",
        }
        cc = tt._container_config_from_config(config)
        assert cc["container_cpu"] == 2
        assert cc["testbox_api_key"] == "secret"
        assert cc["testbox_region"] == "eu-west"

    def test_known_keys_still_defaulted(self):
        import tools.terminal_tool_backends as tt

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


def test_configured_values_reach_registered_provider(monkeypatch, tmp_path):
    from tools import terminal_tool as terminal
    from tools.terminal_tool_lifecycle import _create_configured_env
    from tools.terminal_scope import install_profile_terminal_scope, reset_terminal_scope
    (tmp_path / 'config.yaml').write_text('terminal:\n  backend: testbox\n  container_cpu: 3\n')
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    received = []
    def create(self, **kwargs):
        received.append(kwargs['container_config'])
        return _Env()
    _register(create_environment=create)
    token = install_profile_terminal_scope(tmp_path)
    try:
        config = terminal._get_env_config()
        _create_configured_env(config, config['env_type'], image=None, cwd=config['cwd'],
                               timeout=config['timeout'], task_id='fixture', host_cwd=None)
        assert received[0]['container_cpu'] == 3
    finally:
        reset_terminal_scope(token)


def test_provider_specific_values_survive_factory_shaping():
    from tools.terminal_tool_lifecycle import _create_configured_env
    received = []
    def create(self, **kwargs):
        received.append(kwargs['container_config'])
        return _Env()
    _register(create_environment=create)
    _create_configured_env({'testbox_region': 'eu-west'}, 'testbox', image=None,
                           cwd='/workspace', timeout=30, task_id='fixture', host_cwd=None)
    assert received[0]['testbox_region'] == 'eu-west'


@pytest.mark.parametrize('ambient,scoped', [('local', 'testbox'), ('testbox', 'local')])
def test_reverse_cache_mapping_uses_profile_scope(monkeypatch, tmp_path, ambient, scoped):
    from tools import credential_files as cf
    from tools.terminal_scope import set_terminal_scope, reset_terminal_scope
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('TERMINAL_ENV', ambient)
    _register(cache_path_base='/root/.hermes')
    host = str(tmp_path / 'cache' / 'images' / 'fixture.png')
    token = set_terminal_scope({'TERMINAL_ENV': scoped})
    try:
        visible = cf.to_agent_visible_cache_path(host)
        assert (visible != host) is (scoped == 'testbox')
        assert cf.from_agent_visible_cache_path(visible) == host
    finally:
        reset_terminal_scope(token)


def test_raising_provider_cache_property_returns_path_unchanged(monkeypatch):
    from tools import credential_files as cf
    def broken(self):
        raise RuntimeError('broken provider')
    _register(cache_path_base=property(broken))
    monkeypatch.setenv('TERMINAL_ENV', 'testbox')
    assert cf.from_agent_visible_cache_path('/root/.hermes/cache/images/x') == '/root/.hermes/cache/images/x'
