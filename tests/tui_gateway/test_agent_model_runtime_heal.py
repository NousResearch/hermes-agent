"""Regression: a resumed model row persisted as ``provider="custom"`` + ``base_url`` heals back to
its named local runtime, and the healed identity must NOT be handed the explicit base_url.

Forwarding it demotes the local-runtime rung to the direct-alias rung and the generated endpoint
key is dropped, so the first API turn of a new session answers 401 Invalid API Key. The
session-restore path (``_stored_session_runtime_overrides``) already drops the base_url for the
same reason; this pins the new-agent path (``_resolve_agent_model_runtime``) to the same contract.

The recovery itself (``canonical_custom_identity`` -> a configured ``custom_providers``/``providers``
entry by base_url) is exercised for real against an isolated ``HERMES_HOME``; only the downstream
``_resolve_runtime_with_fallback`` is stubbed so the test observes the routing decision without a
live model server.

Related: #144437, #120833.
"""
from __future__ import annotations

import pytest

from tui_gateway import agent_factory


# An isolated config with ONE custom endpoint, so ``canonical_custom_identity`` can recover a named
# identity from a matching base_url (and must fail to recover for a non-matching one).
_CONFIG_YAML = (
    "model:\n"
    "  provider: custom\n"
    "  model: probe-model\n"
    "custom_providers:\n"
    "  - name: localtest\n"
    "    base_url: http://127.0.0.1:18999/v1\n"
)


class _CapturedFallback:
    """Records the ``resolve_kwargs`` handed to ``_resolve_runtime_with_fallback`` and returns a
    minimal non-fallback resolution, so the test observes the routing decision without a live model
    server."""

    def __init__(self) -> None:
        self.kwargs = None

    def __call__(self, resolve_kwargs):
        self.kwargs = dict(resolve_kwargs)
        resolution = _StubResolution(model=resolve_kwargs.get("target_model") or "")
        return resolution


class _StubResolution:
    """Just the attributes ``_resolve_agent_model_runtime`` reads off the resolution."""

    def __init__(self, model: str) -> None:
        self.used_fallback = False
        self.selected_model = model
        self.runtime = {}


@pytest.fixture
def isolated_custom_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text(_CONFIG_YAML, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.delenv("HERMES_TUI_PROVIDER", raising=False)
    return home


def test_healed_identity_does_not_forward_explicit_base_url(isolated_custom_home, monkeypatch):
    """Recovery succeeds (the persisted base_url matches a configured entry): the healed identity is
    resolved WITHOUT ``explicit_base_url``, so the local-runtime rung keeps its generated key.

    Red on the pre-fix code, where the base_url was forwarded unconditionally and the identity was
    demoted to the direct-alias rung (the 401)."""
    cap = _CapturedFallback()
    monkeypatch.setattr(agent_factory, "_resolve_runtime_with_fallback", cap)
    model_override = {
        "model": "probe-model",
        "provider": "custom",
        "base_url": "http://127.0.0.1:18999/v1",
    }

    agent_factory._resolve_agent_model_runtime(model_override, None)

    assert cap.kwargs is not None, "resolver was never reached"
    # Recovery produced the owning entry's identity, not the opaque "custom".
    assert cap.kwargs["requested"] == "custom:localtest"
    # THE BUG: forwarding base_url to a healed identity demotes it to the direct-alias rung and the
    # generated endpoint key is lost (401 Invalid API Key on the first turn).
    assert "explicit_base_url" not in cap.kwargs


def test_unrecovered_custom_still_forwards_explicit_base_url(isolated_custom_home, monkeypatch):
    """Recovery fails (the base_url matches no entry and the model is unknown): base_url is still
    handed to the direct-alias rung so pool/env credentials can resolve. Guards the recovery-failure
    branch of the fix (green both before and after)."""
    cap = _CapturedFallback()
    monkeypatch.setattr(agent_factory, "_resolve_runtime_with_fallback", cap)
    model_override = {
        "model": "totally-unknown-model",
        "provider": "custom",
        "base_url": "http://127.0.0.1:18777/v1",
    }

    agent_factory._resolve_agent_model_runtime(model_override, None)

    assert cap.kwargs is not None
    assert cap.kwargs["requested"] == "custom"
    assert cap.kwargs["explicit_base_url"] == "http://127.0.0.1:18777/v1"
