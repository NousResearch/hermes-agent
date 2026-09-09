"""Regression: sandbox builder dispatch must not double-bind ``env_type``.

https://github.com/NousResearch/hermes-agent/issues/105414

``_create_environment()`` calls every backend builder with ``env_type=`` passed by keyword, but the
sandbox builders (singularity/daytona/vercel_sandbox) were previously defined via
``functools.partial(_build_sandbox_env, "singularity")`` — a *positional* binding of the same
argument. Any sandbox terminal backend therefore crashed with::

    TypeError: _build_sandbox_env() got multiple values for argument 'env_type'

before the environment object was ever constructed. The builders now share the dispatcher's
keyword-only signature, so this exercises the real dispatch path end to end.
"""

import pytest

import tools.terminal_tool_backends as backends

SANDBOX_TYPES = ("singularity", "daytona", "vercel_sandbox")


class _FakeSandboxEnv:
    """Stands in for Singularity/Daytona/VercelSandbox (no SDK or container binary needed)."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def fake_sandbox_rows(monkeypatch):
    """Point every sandbox row at the fake env so dispatch runs without real backends."""
    for env_type in SANDBOX_TYPES:
        monkeypatch.setitem(backends._SANDBOX_ROWS, env_type,
                            (lambda: _FakeSandboxEnv, True, lambda cc, kw: {}))
    yield


@pytest.mark.parametrize("env_type", SANDBOX_TYPES)
def test_sandbox_builder_dispatch_accepts_env_type_keyword(monkeypatch, fake_sandbox_rows, env_type):
    """Regression: dispatching to a sandbox backend must not raise ``TypeError`` (env_type passed
    twice). The environment should be constructed with the dispatcher's kwargs intact."""
    env = backends._create_environment(env_type=env_type, image="img", cwd="/tmp", timeout=5)

    assert isinstance(env, _FakeSandboxEnv)
    assert env.kwargs["cwd"] == "/tmp"
    assert env.kwargs["timeout"] == 5
    assert env.kwargs["image"] == "img"


def test_local_builder_unaffected(monkeypatch):
    """The non-sandbox builders keep working through the same dispatcher."""
    env = backends._create_environment(env_type="local", image="img", cwd="/tmp", timeout=5)
    assert env is not None
