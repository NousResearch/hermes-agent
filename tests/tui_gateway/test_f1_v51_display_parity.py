"""F1 v5.1 — tools.show / tools.list display-path parity tests.

The display layer must agree with the model-facing schema:
  1. Lazy session (agent None) with a non-empty pin shows EXACTLY the pinned
     pre-assembly inventory — never the broad catalog.
  2. An explicit [] pin shows no enabled toolsets and no tools.
  3. A composite pin plus tool-level denial hides the denied tool.
  4. No-session, lazy-session, and built-agent reporting agree.

Each case runs in a subprocess: re-importing the gateway stack under a temp
HERMES_HOME mutates sys.modules/registry state that must not leak into other
tests sharing the pytest process.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

_CHILD = r'''
import json, os, sys

home = sys.argv[1]
os.environ["HERMES_HOME"] = home
os.environ.pop("HERMES_TUI_TOOLSETS", None)
sys.path.insert(0, os.getcwd())

import tui_gateway.methods_tools as mt
from model_tools import get_tool_definitions

def defs(scope, disabled):
    return sorted({
        t["function"]["name"]
        for t in get_tool_definitions(
            enabled_toolsets=scope, disabled_toolsets=disabled,
            quiet_mode=True, skip_tool_search_assembly=True,
        )
    })

out = {}
case = sys.argv[2]
if case == "lazy_pinned":
    scope, disabled = mt._session_effective_tool_scope({"agent": None, "session_id": "lazy-1"})
    out = {"scope": sorted(scope), "disabled": sorted(disabled), "names": defs(scope, disabled)}
elif case == "empty_pin":
    scope, disabled = mt._session_effective_tool_scope(None)
    out = {"scope": scope, "names": defs(scope, disabled)}
elif case == "composite_denial":
    scope, disabled = mt._session_effective_tool_scope({"agent": None})
    out = {"scope": sorted(scope), "disabled": sorted(disabled), "names": defs(scope, disabled)}
elif case == "parity":
    import hermes_cli.tools_config as tc
    from hermes_cli.config import load_config
    import tui_gateway.server as server

    cfg = load_config()
    builder_scope = sorted(tc._get_platform_tools(cfg, "cli", include_default_mcp_servers=True))
    builder_disabled = sorted(server._resolve_disabled_toolsets(cfg))
    no_session, d1 = mt._session_effective_tool_scope(None)
    lazy, d2 = mt._session_effective_tool_scope({"agent": None, "session_id": "s"})

    class FakeAgent:
        enabled_toolsets = list(builder_scope)
        disabled_toolsets = list(builder_disabled)

    built, d3 = mt._session_effective_tool_scope({"agent": FakeAgent()})
    out = {
        "builder_scope": builder_scope,
        "builder_disabled": builder_disabled,
        "no_session": sorted(no_session), "d1": sorted(d1),
        "lazy": sorted(lazy), "d2": sorted(d2),
        "built": sorted(built), "d3": sorted(d3),
    }
elif case == "empty_agent_scope":
    class FakeAgent:
        enabled_toolsets = []
    scope, _ = mt._session_effective_tool_scope({"agent": FakeAgent()})
    out = {"scope": scope}
print("CHILD_JSON=" + json.dumps(out))
'''


def _run_child(home: Path, case: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(home), case],
        capture_output=True, text=True, cwd=REPO, timeout=300,
        env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    lines = [l for l in proc.stderr.splitlines() if l.startswith("CHILD_JSON=")]
    if not lines:
        raise AssertionError(f"child failed (case={case}):\n{proc.stdout[-500:]}\n{proc.stderr[-800:]}")
    return json.loads(lines[-1].split("=", 1)[1])


def _write_config(home: Path, body: str) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(body)
    return home


def test_lazy_session_nonempty_pin_shows_pinned_inventory(tmp_path):
    """Lazy session (agent None) must resolve the PINNED profile surface — the
    display must never fall back to the broad catalog (the v5.1 bug)."""
    home = _write_config(
        tmp_path / "smokey-pin",
        "tools:\n"
        "  enabled_toolsets: [web, file_readonly, skills_readonly, memory, session_search, clarify]\n"
        "agent:\n"
        "  disabled_toolsets: [cronjob]\n",
    )
    out = _run_child(home, "lazy_pinned")
    assert out["scope"] == [
        "clarify", "file_readonly", "memory", "session_search",
        "skills_readonly", "web",
    ]
    assert "cronjob" in out["disabled"]
    assert "cronjob_manage" not in out["names"]
    assert "terminal" not in out["names"] and "write_file" not in out["names"]
    assert "read_file" in out["names"] and "web_search" in out["names"]


def test_explicit_empty_pin_shows_nothing_enabled(tmp_path):
    """An explicit [] pin must display no enabled toolsets and no tools —
    `enabled if enabled else True` previously showed everything."""
    home = _write_config(
        tmp_path / "empty-pin",
        "tools:\n  enabled_toolsets: []\nagent:\n  disabled_toolsets: [cronjob]\n",
    )
    out = _run_child(home, "empty_pin")
    assert out["scope"] == []
    assert out["names"] == []


def test_composite_pin_with_tool_level_denial_hides_denied_tool(tmp_path):
    """`coding` composite + tool-level `terminal` denial: tools.show must not
    advertise terminal (previously the display ignored disabled_toolsets)."""
    home = _write_config(
        tmp_path / "denial-pin",
        "tools:\n  enabled_toolsets: [coding]\n"
        "agent:\n  disabled_toolsets: [cronjob, terminal, process_manage]\n",
    )
    out = _run_child(home, "composite_denial")
    assert "coding" in out["scope"]
    assert "terminal" in out["disabled"] and "process_manage" in out["disabled"]
    assert "terminal" not in out["names"]
    assert "process_manage" not in out["names"]
    assert "read_file" in out["names"]  # rest of the composite survives


def test_no_session_lazy_and_built_agent_parity(tmp_path):
    """No-session, lazy-session, and built-agent (with the scope the builder
    passes) must all report the same effective scope."""
    home = _write_config(
        tmp_path / "parity",
        "tools:\n"
        "  enabled_toolsets: [web, file_readonly, skills_readonly, memory, session_search, clarify]\n"
        "agent:\n  disabled_toolsets: [cronjob]\n",
    )
    out = _run_child(home, "parity")
    assert out["no_session"] == out["builder_scope"]
    assert out["lazy"] == out["builder_scope"]
    assert out["built"] == out["builder_scope"]
    assert out["d1"] == out["builder_disabled"]
    assert out["d2"] == out["builder_disabled"]
    assert out["d3"] == out["builder_disabled"]


def test_built_agent_intentional_empty_scope_not_reopened(tmp_path):
    """A built agent with an intentional [] keeps [] in the display paths."""
    home = _write_config(
        tmp_path / "agent-empty",
        "tools:\n  enabled_toolsets: [web]\nagent:\n  disabled_toolsets: [cronjob]\n",
    )
    out = _run_child(home, "empty_agent_scope")
    assert out["scope"] == []


# ─── v5.1r2: handler-level (RPC) tests — real tools.show / tools.list paths ──

_HANDLER_CHILD = r'''
import json, os, sys

home = sys.argv[1]
case = sys.argv[2]
os.environ["HERMES_HOME"] = home
# r7: HERMES_DESKTOP is deliberately STRIPPED so the test proves the session
# source is the platform authority — inheriting the parent's env would mask
# the bug this regression guards (the child may run inside the desktop app).
os.environ.pop("HERMES_DESKTOP", None)
os.environ.pop("HERMES_TUI_TOOLSETS", None)
sys.path.insert(0, os.getcwd())

import tui_gateway.server as server
import tui_gateway.methods_tools as mt  # noqa: F401 — registers handlers

def rpc(name, params):
    return server._methods[name]("rid-1", params)

if case == "surface_source":
    # Prince r7 blocker: an unpinned lazy session sourced from "desktop" must
    # resolve the DESKTOP surface (desktop_ui present) even with HERMES_DESKTOP
    # unset — the session source, not the process env, is the platform
    # authority. Contrast: the same profile via a CLI-sourced session must NOT
    # have desktop_ui.
    server._sessions["sd"] = {"agent": None, "session_id": "sd", "source": "desktop"}
    server._sessions["sc"] = {"agent": None, "session_id": "sc", "source": "cli"}

    def enabled_set(sid):
        lst = rpc("tools.list", {"session_id": sid})
        return sorted(i["name"] for i in lst["result"]["toolsets"] if i["enabled"])

    def show_names(sid):
        show = rpc("tools.show", {"session_id": sid})
        return sorted({t["name"] for s in show["result"]["sections"] for t in s["tools"]})

    desktop_enabled = enabled_set("sd")
    cli_enabled = enabled_set("sc")
    desktop_show = show_names("sd")
    print("CHILD_JSON=" + json.dumps({
        "hermes_desktop_env_set": bool(os.environ.get("HERMES_DESKTOP")),
        "desktop_enabled": desktop_enabled,
        "cli_enabled": cli_enabled,
        "desktop_ui_in_desktop": "desktop_ui" in desktop_enabled,
        "desktop_ui_in_cli": "desktop_ui" in cli_enabled,
        "read_terminal_in_desktop_show": "read_terminal" in desktop_show,
    }))
    print("CHILD_DONE", file=sys.stderr)

if case == "built_agent_denial":
    # Built agent carrying its own disabled_toolsets (tool-level denials):
    # tools.show must hide the denied tool from the advertised inventory.
    from model_tools import get_tool_definitions

    class FakeAgent:
        enabled_toolsets = ["coding"]
        disabled_toolsets = ["cronjob", "terminal", "process_manage"]

    server._sessions["s1"] = {"agent": FakeAgent(), "session_id": "s1"}
    show = rpc("tools.show", {"session_id": "s1"})
    names = sorted({
        t["name"] for s in (show["result"]["sections"]) for t in s["tools"]
    })
    lst = rpc("tools.list", {"session_id": "s1"})
    term_entry = next(
        (i for i in lst["result"]["toolsets"] if i["name"] == "terminal"), {}
    )
    print("CHILD_JSON=" + json.dumps({
        "names": names,
        "terminal_entry": term_entry,
        "show_total": show["result"]["total"],
    }))
elif case == "agent_denial_clean_config":
    # Prince r4 blocker: NO config denials — the ONLY denials live on the
    # built agent's disabled_toolsets. Both handlers must honor the agent's
    # runtime scope (the early-return on enabled_toolsets used to skip the
    # agent-disabled read entirely, so the display leaked denied tools).
    class AgentOnlyDenials:
        enabled_toolsets = ["coding"]
        disabled_toolsets = ["terminal", "process_manage"]

    server._sessions["s1"] = {"agent": AgentOnlyDenials(), "session_id": "s1"}
    show = rpc("tools.show", {"session_id": "s1"})
    names = sorted({
        t["name"] for s in (show["result"]["sections"]) for t in s["tools"]
    })
    lst = rpc("tools.list", {"session_id": "s1"})
    term_entry = next(
        (i for i in lst["result"]["toolsets"] if i["name"] == "terminal"), {}
    )
    print("CHILD_JSON=" + json.dumps({
        "names": names,
        "terminal_entry": term_entry,
        "show_total": show["result"]["total"],
    }))
elif case == "cross_profile_lazy":
    # Lazy session bound to profile B while the process-level profile differs:
    # tools.show must resolve profile B's pin, not the process-level config.
    other_home = sys.argv[3]
    server._sessions["s2"] = {
        "agent": None,
        "session_id": "s2",
        "profile_home": other_home,
    }
    show = rpc("tools.show", {"session_id": "s2"})
    names = sorted({
        t["name"] for s in (show["result"]["sections"]) for t in s["tools"]
    })
    print("CHILD_JSON=" + json.dumps({"names": names}))
elif case == "parity_exact":
    # No-session vs lazy-session: exact same inventory from both RPC handlers.
    no_sess_show = rpc("tools.show", {})
    no_sess_list = rpc("tools.list", {})
    server._sessions["s3"] = {"agent": None, "session_id": "s3"}
    lazy_show = rpc("tools.show", {"session_id": "s3"})
    lazy_list = rpc("tools.list", {"session_id": "s3"})

    def show_names(r):
        return sorted({t["name"] for s in r["result"]["sections"] for t in s["tools"]})

    def list_enabled(r):
        return sorted(i["name"] for i in r["result"]["toolsets"] if i["enabled"])

    print("CHILD_JSON=" + json.dumps({
        "no_sess_show": show_names(no_sess_show),
        "lazy_show": show_names(lazy_show),
        "no_sess_list": list_enabled(no_sess_list),
        "lazy_list": list_enabled(lazy_list),
        "no_sess_total": no_sess_show["result"]["total"],
        "lazy_total": lazy_show["result"]["total"],
    }))
elif case == "unpinned_denial_parity":
    # Prince r5 blocker: NO pin (unpinned profile) + cronjob/tool-level
    # denials. tools.list must advertise the SAME filtered inventory as
    # tools.show — the raw toolset info must never re-advertise denied tools.
    server._sessions["su"] = {"agent": None, "session_id": "su"}
    lst = rpc("tools.list", {"session_id": "su"})
    show = rpc("tools.show", {"session_id": "su"})

    # Union of every tool advertised by tools.list's per-toolset lists.
    list_tools = sorted({
        t for i in lst["result"]["toolsets"] if i["enabled"]
        for t in (i["tools"] or [])
    })
    show_names = sorted({
        t["name"] for s in show["result"]["sections"] for t in s["tools"]
    })
    cronjob_entry = next(
        (i for i in lst["result"]["toolsets"] if i["name"] == "cronjob"), {}
    )
    # r6: a composite (coding) with only tool-level children denied must STAY
    # enabled — the flag distinguishes explicit toolset denial from child
    # filtering.
    coding_entry = next(
        (i for i in lst["result"]["toolsets"] if i["name"] == "coding"), {}
    )
    print("CHILD_JSON=" + json.dumps({
        "list_tools": list_tools,
        "show_names": show_names,
        "cronjob_entry": cronjob_entry,
        "coding_entry": coding_entry,
        "file_toolsets": [
            i for i in lst["result"]["toolsets"] if i["name"] in {"file", "coding"}
        ],
        "cronjob_manage_in_list_tools": "cronjob_manage" in list_tools,
    }))
print("CHILD_DONE", file=sys.stderr)
'''

def _run_handler_child(home: Path, case: str, other_home: str | None = None) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", _HANDLER_CHILD, str(home), case, other_home or ""],
        capture_output=True, text=True, cwd=REPO, timeout=300,
        env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    lines = [l for l in proc.stderr.splitlines() if l.startswith("CHILD_JSON=")]
    if not lines:
        raise AssertionError(f"child failed (case={case}):\n{proc.stdout[-400:]}\n{proc.stderr[-900:]}")
    return json.loads(lines[-1].split("=", 1)[1])


def test_rpc_built_agent_tool_denials_hidden_from_display(tmp_path):
    """Handler-level: a built agent's tool-level denials hide `terminal` and
    `process_manage` from BOTH tools.show and the terminal toolset entry in
    tools.list (previously tools.list showed unfiltered resolved_tools)."""
    home = _write_config(
        tmp_path / "denial",
        "tools:\n  enabled_toolsets: [coding]\n"
        "agent:\n  disabled_toolsets: [cronjob, terminal, process_manage]\n",
    )
    out = _run_handler_child(home, "built_agent_denial")
    assert "terminal" not in out["names"]
    assert "process_manage" not in out["names"]
    assert "read_file" in out["names"]  # composite rest survives
    assert "cronjob_manage" not in out["names"]
    term = out["terminal_entry"]
    # `terminal` was disabled at TOOL level inside the enabled `coding`
    # composite: the toolset entry must show the denial — either the toolset
    # resolves empty (enabled=False once its only tools are stripped) or its
    # advertised tools must omit the denied names. Both are honest displays;
    # the dishonest one (enabled + full unfiltered list) is what we forbid.
    if term.get("enabled"):
        assert "terminal" not in (term.get("tools") or [])
        assert "process_manage" not in (term.get("tools") or [])
    else:
        assert not (term.get("tools") or [])


def test_rpc_agent_only_denials_honored_with_clean_config(tmp_path):
    """Prince r4 blocker: config has NO denials — the ONLY denials live on the
    built agent's `disabled_toolsets`. Both handlers must honor the agent's
    runtime scope; the r3 early-return on `enabled_toolsets` skipped the
    agent-disabled read, leaking denied tools into the display."""
    home = _write_config(
        tmp_path / "clean-config",
        "tools:\n  enabled_toolsets: [coding]\nagent:\n  disabled_toolsets: [cronjob]\n",
    )
    out = _run_handler_child(home, "agent_denial_clean_config")
    assert "terminal" not in out["names"]
    assert "process_manage" not in out["names"]
    assert "read_file" in out["names"] and "web_search" in out["names"]
    term = out["terminal_entry"]
    if term.get("enabled"):
        assert "terminal" not in (term.get("tools") or [])
        assert "process_manage" not in (term.get("tools") or [])
    else:
        assert not (term.get("tools") or [])


def test_rpc_unpinned_profile_denial_parity_between_list_and_show(tmp_path):
    """Prince r5 blocker: an UNPINNED profile with cronjob + tool-level
    denials must get the SAME filtered inventory from tools.list and
    tools.show — `enabled_set is None` previously bypassed the filter and
    re-advertised denied tools from raw toolset info."""
    home = _write_config(
        tmp_path / "unpinned-denials",
        # NO tools.enabled_toolsets key at all — genuinely unpinned. Denies
        # the cronjob TOOLSET explicitly plus tool-level denials inside the
        # enabled `file` toolset.
        "agent:\n  disabled_toolsets: [cronjob, terminal, process_manage, write_file, patch]\n",
    )
    out = _run_handler_child(home, "unpinned_denial_parity")
    # tools.list's advertised union must not contain denied tools.
    assert "cronjob_manage" not in out["list_tools"]
    assert "terminal" not in out["list_tools"]
    assert "process_manage" not in out["list_tools"]
    assert "write_file" not in out["list_tools"]
    assert "patch" not in out["list_tools"]
    # The model-facing schema (tools.show) must not contain them either.
    assert "cronjob_manage" not in out["show_names"]
    assert "terminal" not in out["show_names"]
    assert "process_manage" not in out["show_names"]
    assert "write_file" not in out["show_names"]
    assert "patch" not in out["show_names"]
    # Denial parity is the contract; exact set equality is NOT, for two
    # legitimate axes: (1) tools.show is check_fn-filtered (headless browser/
    # vision go missing), and (2) tools.list's static resolution excludes
    # registry-registered tools (e.g. apply_layout in desktop_ui) that
    # get_tool_definitions includes. Neither axis can leak a DENIED tool.
    #
    # The two sentinel assertions above-document; the real cross-axis guard:
    # every tool in show but not in list must belong to a registry-added or
    # availability-gated class — assert the delta contains no denied tool
    # (already asserted above) and no pin-listed tool could appear there.
    # The cronjob toolset entry must be honest about its filtering.
    cron = out["cronjob_entry"]
    # Prince r6: an EXPLICITLY disabled toolset name reports enabled=False —
    # an emptied list with enabled=true was the r5 reporting inconsistency.
    assert cron.get("enabled") is False
    assert not (cron.get("tools") or [])
    # r6 symmetric case: a toolset IN the resolved scope with only tool-level
    # child denials stays enabled with those children filtered from its
    # advertised list. The config also denies `write_file`/`patch` by tool
    # name inside the enabled `file` toolset: `file` stays enabled=True and
    # its advertised list drops exactly those children. (`coding` is a
    # posture toolset — session-selected, and legitimately NOT in an
    # unpinned profile's resolved scope, so its enabled=False is the scope's
    # answer, not a denial artifact.)
    file_entry = next(
        (i for i in out["file_toolsets"] if i["name"] == "file"), {}
    )
    assert file_entry.get("enabled") is True
    file_tools = set(file_entry.get("tools") or [])
    assert "read_file" in file_tools and "search_files" in file_tools
    assert "write_file" not in file_tools and "patch" not in file_tools


def test_rpc_cross_profile_lazy_session_resolves_bound_profile(tmp_path):
    """Handler-level: a lazy session bound to profile_home=B must display B's
    pin even though the process-level config differs (no cross-profile leak)."""
    home_a = _write_config(
        tmp_path / "profile-a",
        "tools:\n  enabled_toolsets: [web]\nagent:\n  disabled_toolsets: [cronjob]\n",
    )
    home_b = _write_config(
        tmp_path / "profile-b",
        "tools:\n  enabled_toolsets: [terminal, file]\nagent:\n  disabled_toolsets: [cronjob]\n",
    )
    # Process-level profile is A; the lazy session is bound to B.
    out = _run_handler_child(home_a, "cross_profile_lazy", str(home_b))
    # B's pin admits terminal+file; A's pin does not — the display must
    # reflect B (the session's own profile).
    assert "terminal" in out["names"]
    assert "write_file" in out["names"]
    assert "read_file" in out["names"]
    # A-only surface must not leak through (B denies nothing A has beyond
    # pins, but web is NOT in B's pin so it must be absent).
    assert "web_search" not in out["names"]


def test_rpc_no_session_and_lazy_session_exact_parity(tmp_path):
    """Handler-level: no-session and lazy-session report EXACTLY the same
    inventory from both tools.show (names + total) and tools.list (enabled
    toolset set)."""
    home = _write_config(
        tmp_path / "parity",
        "tools:\n"
        "  enabled_toolsets: [web, file_readonly, skills_readonly, memory, session_search, clarify]\n"
        "agent:\n  disabled_toolsets: [cronjob]\n",
    )
    out = _run_handler_child(home, "parity_exact")
    assert out["no_sess_show"] == out["lazy_show"]
    assert out["no_sess_list"] == out["lazy_list"]
    assert out["no_sess_total"] == out["lazy_total"]
    assert "cronjob" not in out["no_sess_list"]


def test_rpc_surface_resolves_from_session_source_not_process_env(tmp_path):
    """Prince r7 blocker: an unpinned lazy session sourced from "desktop" must
    resolve the DESKTOP surface (desktop_ui present) even with HERMES_DESKTOP
    unset — remote/URL gateways run without that env, so the session's own
    source is the platform authority. Contrast: a CLI-sourced session on the
    same process must not gain the desktop surface."""
    home = _write_config(tmp_path / "unpinned", "agent:\n  disabled_toolsets: [cronjob]\n")
    out = _run_handler_child(home, "surface_source")
    assert out["hermes_desktop_env_set"] is False  # env was NOT the authority
    assert out["desktop_ui_in_desktop"] is True  # desktop surface for desktop source
    assert out["desktop_ui_in_cli"] is False  # no leak into the CLI session
    assert out["read_terminal_in_desktop_show"] is True  # real desktop_ui tool in show
    assert out["desktop_enabled"] != out["cli_enabled"]  # genuinely per-source


_CONCURRENT_CHILD = r'''
import json, os, sys, threading

home_a, home_b = sys.argv[1], sys.argv[2]
os.environ["HERMES_HOME"] = home_a  # process-level = profile A
os.environ.pop("HERMES_TUI_TOOLSETS", None)
sys.path.insert(0, os.getcwd())

import tui_gateway.server as server

# Two lazy sessions bound to different profiles; hammer both handlers
# concurrently from many threads and collect every inventory each saw.
server._sessions["sa"] = {"agent": None, "session_id": "sa", "profile_home": home_a}
server._sessions["sb"] = {"agent": None, "session_id": "sb", "profile_home": home_b}

results = {"a": set(), "b": set(), "wrong": []}
lock = threading.Lock()

def hammer(sid, key, expected_has, expected_absent, n=40):
    for _ in range(n):
        show = server._methods["tools.show"]("rid", {"session_id": sid})
        names = frozenset(
            t["name"] for s in show["result"]["sections"] for t in s["tools"]
        )
        with lock:
            results[key].add(frozenset(names))
            if expected_has and not (expected_has & names):
                results["wrong"].append({key, "missing-expected"})
            if expected_absent & names:
                results["wrong"].append({key, "leaked-denied"})

threads = [
    threading.Thread(target=hammer, args=("sa", "a", {"web_search"}, {"terminal"})),
    threading.Thread(target=hammer, args=("sa", "a", {"web_search"}, {"terminal"})),
    threading.Thread(target=hammer, args=("sb", "b", {"terminal"}, {"web_search"})),
    threading.Thread(target=hammer, args=("sb", "b", {"terminal"}, {"web_search"})),
]
for t in threads: t.start()
for t in threads: t.join()

print("CHILD_JSON=" + json.dumps({
    "a_variants": len(results["a"]),
    "b_variants": len(results["b"]),
    "races": results["wrong"],
    "a_names": sorted(next(iter(results["a"]))) if len(results["a"]) == 1 else None,
    "b_names": sorted(next(iter(results["b"]))) if len(results["b"]) == 1 else None,
}))
'''


def test_rpc_concurrent_cross_profile_handlers_never_race(tmp_path):
    """Handler-level concurrency: many threads hammer tools.show for lazy
    sessions bound to two DIFFERENT profiles simultaneously. The ContextVar
    scoping must keep every profile-A answer A-pinned and every profile-B
    answer B-pinned — no cross-profile leakage, no flapping inventories."""
    home_a = _write_config(
        tmp_path / "profile-a",
        "tools:\n  enabled_toolsets: [web]\nagent:\n  disabled_toolsets: [cronjob]\n",
    )
    home_b = _write_config(
        tmp_path / "profile-b",
        "tools:\n  enabled_toolsets: [terminal, file]\nagent:\n  disabled_toolsets: [cronjob]\n",
    )
    proc = subprocess.run(
        [sys.executable, "-c", _CONCURRENT_CHILD, str(home_a), str(home_b)],
        capture_output=True, text=True, cwd=REPO, timeout=600,
        env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    lines = [l for l in proc.stderr.splitlines() if l.startswith("CHILD_JSON=")]
    if not lines:
        raise AssertionError(f"child failed:\n{proc.stdout[-400:]}\n{proc.stderr[-900:]}")
    out = json.loads(lines[-1].split("=", 1)[1])
    assert out["races"] == [], out["races"]
    # Deterministic: each session always resolved to ONE stable inventory.
    assert out["a_variants"] == 1, f"profile A inventory flapped: {out['a_variants']} variants"
    assert out["b_variants"] == 1, f"profile B inventory flapped: {out['b_variants']} variants"
    assert "web_search" in out["a_names"] and "terminal" not in out["a_names"]
    assert "terminal" in out["b_names"] and "web_search" not in out["b_names"]
