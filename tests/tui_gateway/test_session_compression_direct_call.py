"""Split-module leaf helpers must resolve without server.py's rebind.

Regression for #115572: ``_apply_live_compression_config`` referenced
``is_truthy_value`` bare — the name only existed in ``server.py``'s namespace
after ``method_ctx.bind_module`` rebinds the body, so calling the module
function directly raised ``NameError``. The leaf helper must import at module
level like every other statically-resolvable dependency.
"""

from __future__ import annotations

from types import SimpleNamespace

from tui_gateway.session_compression import _apply_live_compression_config


def test_live_compression_config_applies_without_server_rebind():
    agent = SimpleNamespace()
    _apply_live_compression_config(
        agent,
        {"compression": {"enabled": True, "codex_responses_native": "true"}},
    )
    assert agent.compression_enabled is True
    assert agent.codex_responses_native_compaction is True
