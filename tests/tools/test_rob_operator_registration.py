"""Tests for tools.rob_operator_registration — the actual tool-registry
surface Rob is exposed through, as opposed to the underlying functions in
tools.rob_operator_tools (covered separately by
tests/tools/test_rob_operator_tools.py).

An earlier pass registered 20 tools by claim, but no test ever reproduced
that count independently or dispatched every registered handler through
the real registry entry — an independent review had to write a throwaway
script to catch that ``container_exec_readonly`` was registered while
being permanently non-functional (its inner command always fails the
guard's docker allowlist). This file makes that verification permanent
instead of ad hoc.
"""

from __future__ import annotations

from tools import rob_operator_registration  # noqa: F401 — import triggers registry.register calls
from tools.registry import registry

EXPECTED_TOOL_COUNT = 19

# One harmless/invalid argument set per registered tool, used to prove the
# full dispatch path (schema -> handler -> guard/profile -> redaction)
# never raises, without touching any real production resource.
_DISPATCH_ARGS = {
    "rob_docker_ps": {},
    "rob_docker_inspect": {"container": "nonexistent-verification-only"},
    "rob_docker_logs": {"container": "nonexistent-verification-only"},
    "rob_docker_stats": {},
    "rob_docker_network_inspect": {"network": "nonexistent-verification-only"},
    "rob_docker_volume_inspect": {"volume": "nonexistent-verification-only"},
    "rob_docker_compose_ps": {"project_dir": "/tmp"},
    "rob_systemd_status": {"unit": "nonexistent-verification-only.service"},
    "rob_systemd_show": {"unit": "nonexistent-verification-only.service"},
    "rob_journal_query": {"unit": "nonexistent-verification-only.service"},
    "rob_git_inspect": {"repo": ".", "operation": "status"},
    "rob_network_probe": {"host": "127.0.0.1", "port": 1},
    "rob_http_probe": {"url": "http://127.0.0.1:1"},
    "rob_tls_inspect": {"host_name": "127.0.0.1", "port": 1},
    "rob_host_metrics": {},
    "rob_process_inspect": {"pid": 1},
    "rob_db_select": {"profile": "nonexistent-verification-only", "query": "SELECT 1"},
    "rob_schema_inspect": {"profile": "nonexistent-verification-only"},
    "rob_env_presence": {"names": ["PATH"]},
}


def _rob_entries():
    return {e.name: e for e in registry.get_all_entries() if e.toolset == "rob_operator"}


class TestRegisteredToolSurface:
    def test_exact_registered_count(self):
        assert len(_rob_entries()) == EXPECTED_TOOL_COUNT

    def test_every_tool_uses_rob_prefix(self):
        assert all(name.startswith("rob_") for name in _rob_entries())

    def test_every_tool_self_describes_as_read_only(self):
        entries = _rob_entries()
        assert all("[READ-ONLY]" in e.schema["function"]["description"] for e in entries.values())

    def test_schema_name_matches_registration_name(self):
        entries = _rob_entries()
        assert all(name == e.schema["function"]["name"] for name, e in entries.items())

    def test_dispatch_args_defined_for_every_registered_tool(self):
        # If this fails, a tool was registered without a corresponding
        # entry above — the dispatch test below would otherwise silently
        # skip it instead of failing.
        missing = set(_rob_entries()) - set(_DISPATCH_ARGS)
        assert not missing, f"no dispatch args defined for: {missing}"

    def test_container_exec_readonly_is_not_registered(self):
        # Regression guard: this tool is permanently denied by the guard's
        # docker allowlist (no `exec` entry) and was deliberately removed
        # from the registry rather than left as an always-failing tool
        # call. If this ever starts passing as "registered", the guard's
        # docker allowlist must have grown an `exec` entry — which is a new
        # capability, not a bugfix, and needs its own explicit review.
        assert "rob_container_exec_readonly" not in _rob_entries()


class TestDispatchLevel:
    def test_every_handler_dispatches_without_raising(self):
        failures = []
        for name, entry in _rob_entries().items():
            args = _DISPATCH_ARGS[name]
            try:
                entry.handler(args)
            except Exception as exc:  # noqa: BLE001 — intentionally broad, this IS the assertion
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
        assert not failures, "\n".join(failures)

    def test_db_select_and_schema_inspect_refuse_without_configured_profile(self):
        entries = _rob_entries()
        db_result = entries["rob_db_select"].handler({"profile": "nonexistent-verification-only", "query": "SELECT 1"})
        assert db_result["ok"] is False
        assert "profile" in db_result["error"].lower()

        schema_result = entries["rob_schema_inspect"].handler({"profile": "nonexistent-verification-only"})
        assert schema_result["ok"] is False
