"""The Vercel backend against the real, locked ``vercel`` SDK.

``test_vercel_sandbox_environment.py`` replaces ``vercel.sandbox`` with a fake module, so it cannot
see a name the installed SDK no longer exports or a resource request the SDK refuses. These tests
import the real package and stop only at the network call.
"""

import pytest

sandbox_sdk = pytest.importorskip("vercel.sandbox")

from tools.environments import vercel_sandbox


class _CreateCalled(Exception):
    pass


@pytest.fixture()
def create_calls(monkeypatch):
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        raise _CreateCalled

    # raising=False: on an SDK without Sandbox.create the backend's own failure is what surfaces.
    monkeypatch.setattr(sandbox_sdk.Sandbox, "create", staticmethod(create), raising=False)
    monkeypatch.setattr(vercel_sandbox, "_load_snapshots", dict)
    return calls


@pytest.mark.parametrize(("cpu", "memory", "vcpus"), [(1, 5120, 1), (2, 2048, 2), (4, 0, 4)])
def test_container_resources_reach_sandbox_create(create_calls, cpu, memory, vcpus):
    with pytest.raises(_CreateCalled):
        vercel_sandbox.VercelSandboxEnvironment(cpu=cpu, memory=memory, persistent_filesystem=False)

    resources = create_calls[0]["resources"]
    assert isinstance(resources, sandbox_sdk.Resources)
    assert resources.vcpus == vcpus
    assert resources.memory is None
