#!/usr/bin/env python3
"""Tests for profile-backed delegation in delegate_task (issue #41889).

Profile delegation runs the subagent as an in-process child AIAgent built with
the target profile's SOUL.md + model/provider/credentials + toolsets. These
tests mock the profile-resolution and child-execution layers so no real
AIAgent, CLI, or API calls happen.

Run with:  scripts/run_tests.sh tests/tools/test_delegate_profile.py -q
"""

import json
import threading
import unittest
from unittest.mock import MagicMock, patch

from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    _build_child_system_prompt,
    _build_top_level_description,
    _coerce_profile_memory,
    _resolve_profile_bundle,
    delegate_task,
)


def _make_mock_parent(depth=0):
    parent = MagicMock()
    parent.base_url = "https://api.example/v1"
    parent.api_key = "parent-key"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent._delegate_spinner = None
    parent._memory_manager = None
    parent.session_id = "parent-sess"
    parent._current_turn_id = ""
    parent.session_estimated_cost_usd = 0.0
    return parent


class TestProfileSchema(unittest.TestCase):
    def test_top_level_profile_property(self):
        props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
        self.assertIn("profile", props)
        self.assertEqual(props["profile"]["type"], "string")

    def test_per_task_profile_property(self):
        task_props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"][
            "items"
        ]["properties"]
        self.assertIn("profile", task_props)

    def test_description_mentions_profile(self):
        self.assertIn("profile", _build_top_level_description().lower())

    def test_top_level_profile_memory_property(self):
        props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
        self.assertIn("profile_memory", props)
        self.assertEqual(props["profile_memory"]["enum"], ["read", "write"])

    def test_per_task_profile_memory_property(self):
        task_props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"][
            "items"
        ]["properties"]
        self.assertIn("profile_memory", task_props)
        self.assertEqual(task_props["profile_memory"]["enum"], ["read", "write"])


class TestSoulInjection(unittest.TestCase):
    def test_profile_soul_prepended(self):
        prompt = _build_child_system_prompt(
            "do the thing", profile_soul="I am the Reader. Terse."
        )
        self.assertTrue(prompt.startswith("I am the Reader. Terse."))
        self.assertIn("YOUR TASK:", prompt)

    def test_no_soul_unchanged(self):
        prompt = _build_child_system_prompt("do the thing")
        self.assertTrue(prompt.startswith("You are a focused subagent"))


class TestResolveProfileBundle(unittest.TestCase):
    @patch("hermes_cli.profiles.profile_exists", return_value=False)
    def test_missing_profile_raises(self, _exists):
        with self.assertRaises(ValueError) as ctx:
            _resolve_profile_bundle("ghost")
        self.assertIn("does not exist", str(ctx.exception))

    def test_bundle_fields(self):
        # Patch the dependencies _resolve_profile_bundle imports at call time.
        with patch("hermes_cli.profiles.profile_exists", return_value=True), patch(
            "hermes_cli.profiles.get_profile_dir"
        ) as gpd, patch("hermes_cli.config.load_config") as lc, patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider"
        ) as rrp, patch(
            "hermes_cli.tools_config._get_platform_tools",
            return_value={"web", "file"},
        ):
            fake_dir = MagicMock()
            # SOUL.md path → not a file (skip reading)
            fake_dir.__truediv__.return_value.is_file.return_value = False
            gpd.return_value = fake_dir
            lc.return_value = {"model": {"default": "m/x", "provider": "prov"}}
            rrp.return_value = {
                "provider": "prov",
                "base_url": "https://prov/v1",
                "api_key": "prof-key",
                "api_mode": "chat_completions",
            }
            bundle = _resolve_profile_bundle("reader")
        self.assertEqual(bundle["name"], "reader")
        self.assertEqual(bundle["model"], "m/x")
        self.assertEqual(bundle["api_key"], "prof-key")
        self.assertEqual(bundle["base_url"], "https://prov/v1")
        self.assertEqual(sorted(bundle["toolsets"]), ["file", "web"])

    def test_env_scoped_without_mutating_os_environ(self):
        # Regression guard for the in-process concurrency concern: the
        # profile's .env must reach credential resolution via the secret scope
        # (a contextvar), NOT by mutating os.environ. We assert that during
        # resolve_runtime_provider the profile key is visible through
        # get_secret while os.environ stays untouched.
        import os
        import tempfile
        from pathlib import Path
        from agent.secret_scope import get_secret

        seen = {}
        before = dict(os.environ)

        def _capture(*_a, **_k):
            seen["scope_value"] = get_secret("PROFILE_ONLY_KEY")
            seen["os_environ_value"] = os.environ.get("PROFILE_ONLY_KEY")
            return {
                "provider": "prov",
                "base_url": "u",
                "api_key": "k",
                "api_mode": "chat_completions",
            }

        with tempfile.TemporaryDirectory() as td:
            prof_dir = Path(td)
            (prof_dir / ".env").write_text(
                "PROFILE_ONLY_KEY=secret-from-profile\n", encoding="utf-8"
            )
            with patch(
                "hermes_cli.profiles.profile_exists", return_value=True
            ), patch(
                "hermes_cli.profiles.get_profile_dir", return_value=prof_dir
            ), patch(
                "hermes_cli.config.load_config",
                return_value={"model": {"default": "m", "provider": "prov"}},
            ), patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                side_effect=_capture,
            ), patch(
                "hermes_cli.tools_config._get_platform_tools", return_value=set()
            ):
                _resolve_profile_bundle("reader")

        # The credential was visible through the scope during resolution …
        self.assertEqual(seen["scope_value"], "secret-from-profile")
        # … but never leaked into os.environ, and os.environ is unchanged after.
        self.assertIsNone(seen["os_environ_value"])
        self.assertEqual(dict(os.environ), before)

    def test_bom_prefixed_files_decoded_cleanly(self):
        # Regression: a Windows editor (Notepad) saves .env / SOUL.md with a
        # UTF-8 BOM. A BOM does NOT raise UnicodeDecodeError under plain utf-8 —
        # it decodes to '﻿', which would corrupt the FIRST .env key name
        # (silently dropping that credential) and inject '﻿' as the first
        # character of the persona. utf-8-sig strips it; the latin-1 fallback
        # only fires on cp1252, so it never covered this case.
        import tempfile
        from pathlib import Path
        from agent.secret_scope import get_secret

        seen = {}

        def _capture(*_a, **_k):
            # FIRST key must be intact — not '﻿PROFILE_ONLY_KEY'.
            seen["scope_value"] = get_secret("PROFILE_ONLY_KEY")
            return {"provider": "prov", "base_url": "u", "api_key": "k",
                    "api_mode": "chat_completions"}

        with tempfile.TemporaryDirectory() as td:
            prof_dir = Path(td)
            # encoding="utf-8-sig" writes the BOM, emulating Notepad.
            (prof_dir / ".env").write_text(
                "PROFILE_ONLY_KEY=secret-from-profile\n", encoding="utf-8-sig"
            )
            (prof_dir / "SOUL.md").write_text(
                "You are the reader profile.", encoding="utf-8-sig"
            )
            with patch(
                "hermes_cli.profiles.profile_exists", return_value=True
            ), patch(
                "hermes_cli.profiles.get_profile_dir", return_value=prof_dir
            ), patch(
                "hermes_cli.config.load_config",
                return_value={"model": {"default": "m", "provider": "prov"}},
            ), patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                side_effect=_capture,
            ), patch(
                "hermes_cli.tools_config._get_platform_tools", return_value=set()
            ):
                bundle = _resolve_profile_bundle("reader")

        # The first .env credential survived the BOM …
        self.assertEqual(seen["scope_value"], "secret-from-profile")
        # … and the persona starts at the real first character, no '﻿'.
        self.assertEqual(bundle["soul"], "You are the reader profile.")
        self.assertFalse(bundle["soul"].startswith("﻿"))


class TestDelegateTaskProfileRouting(unittest.TestCase):
    def setUp(self):
        self.parent = _make_mock_parent()

    @patch("tools.delegate_tool._run_single_child")
    @patch("tools.delegate_tool._build_child_agent")
    @patch("tools.delegate_tool._resolve_profile_bundle")
    def test_single_profile_overrides_passed(self, mbundle, mbuild, mrun):
        mbundle.return_value = {
            "name": "reader",
            "soul": "Reader persona",
            "model": "prof/model",
            "provider": "prov",
            "base_url": "https://prov/v1",
            "api_key": "prof-key",
            "api_mode": "chat_completions",
            "toolsets": ["web"],
        }
        fake_child = MagicMock()
        fake_child.model = "prof/model"
        mbuild.return_value = fake_child
        mrun.return_value = {
            "task_index": 0,
            "profile": "reader",
            "status": "completed",
            "summary": "ok",
        }
        out = delegate_task(
            goal="extract key points", profile="reader", parent_agent=self.parent
        )
        data = json.loads(out)
        self.assertEqual(data["results"][0]["profile"], "reader")
        # _build_child_agent received the profile's overrides + soul + name.
        kwargs = mbuild.call_args.kwargs
        self.assertEqual(kwargs["model"], "prof/model")
        self.assertEqual(kwargs["override_provider"], "prov")
        self.assertEqual(kwargs["override_api_key"], "prof-key")
        self.assertEqual(kwargs["override_base_url"], "https://prov/v1")
        self.assertEqual(kwargs["profile_soul"], "Reader persona")
        self.assertEqual(kwargs["profile_name"], "reader")

    @patch("tools.delegate_tool._run_single_child")
    @patch("tools.delegate_tool._build_child_agent")
    @patch("tools.delegate_tool._resolve_profile_bundle")
    def test_top_level_profile_inherited_by_batch_tasks(self, mbundle, mbuild, mrun):
        # delegate_task(profile="reader", tasks=[{...}, {...}]) must apply
        # "reader" to EACH batch item that doesn't override it. Regression
        # guard for the top-level-profile + batch inheritance mismatch.
        mbundle.return_value = {
            "name": "reader",
            "soul": "Reader persona",
            "model": "prof/model",
            "provider": "prov",
            "base_url": "u",
            "api_key": "k",
            "api_mode": "chat_completions",
            "toolsets": None,
        }
        fake_child = MagicMock()
        fake_child.model = "prof/model"
        mbuild.return_value = fake_child
        mrun.return_value = {"task_index": 0, "status": "completed", "summary": "ok"}
        delegate_task(
            profile="reader",
            tasks=[{"goal": "a"}, {"goal": "b"}],
            parent_agent=self.parent,
        )
        # The profile was resolved once per batch task (inherited by both).
        resolved = [c.args[0] for c in mbundle.call_args_list]
        self.assertEqual(resolved, ["reader", "reader"])
        # Every child was built with the profile's soul + name.
        for call in mbuild.call_args_list:
            self.assertEqual(call.kwargs["profile_name"], "reader")
            self.assertEqual(call.kwargs["profile_soul"], "Reader persona")

    @patch("tools.delegate_tool._run_single_child")
    @patch("tools.delegate_tool._build_child_agent")
    @patch("tools.delegate_tool._resolve_profile_bundle")
    def test_per_task_profile_overrides_top_level(self, mbundle, mbuild, mrun):
        # A task's own 'profile' wins over the inherited top-level one.
        mbundle.side_effect = lambda name: {
            "name": name,
            "soul": f"{name} persona",
            "model": "m",
            "provider": "p",
            "base_url": "u",
            "api_key": "k",
            "api_mode": "chat_completions",
            "toolsets": None,
        }
        fake_child = MagicMock()
        fake_child.model = "m"
        mbuild.return_value = fake_child
        mrun.return_value = {"task_index": 0, "status": "completed", "summary": "ok"}
        delegate_task(
            profile="reader",
            tasks=[{"goal": "a"}, {"goal": "b", "profile": "writer"}],
            parent_agent=self.parent,
        )
        resolved = [c.args[0] for c in mbundle.call_args_list]
        self.assertEqual(resolved, ["reader", "writer"])

    @patch(
        "tools.delegate_tool._resolve_profile_bundle",
        side_effect=ValueError("Profile 'ghost' does not exist."),
    )
    def test_invalid_profile_returns_tool_error(self, _mb):
        out = delegate_task(goal="g", profile="ghost", parent_agent=self.parent)
        data = json.loads(out)
        self.assertIn("error", data)
        self.assertIn("does not exist", data["error"])

    @patch("tools.delegate_tool._build_child_agent")
    @patch("tools.delegate_tool._resolve_profile_bundle")
    def test_background_plus_profile_allowed(self, mbundle, mbuild):
        # background+profile must NOT be rejected; it should reach async dispatch.
        mbundle.return_value = {
            "name": "reader",
            "soul": "",
            "model": "prof/model",
            "provider": "prov",
            "base_url": "u",
            "api_key": "k",
            "api_mode": "chat_completions",
            "toolsets": None,
        }
        fake_child = MagicMock()
        fake_child.model = "prof/model"
        mbuild.return_value = fake_child
        with patch(
            "tools.async_delegation.dispatch_async_delegation",
            return_value={"status": "dispatched", "delegation_id": "d1"},
        ), patch("tools.approval.get_current_session_key", return_value=""):
            out = delegate_task(
                goal="g", profile="reader", background=True, parent_agent=self.parent
            )
        data = json.loads(out)
        # Crucially NOT a rejection that background can't be combined with profile.
        self.assertNotIn("cannot be combined", json.dumps(data))
        self.assertEqual(data.get("status"), "dispatched")


class TestProfileToolsetBounding(unittest.TestCase):
    """A profile's toolset preferences are bounded by the parent's tools
    (least privilege), but the narrowing must not be silent: tools the parent
    can't grant are recorded on the child and surfaced in the result.
    """

    def _parent(self, enabled):
        from types import SimpleNamespace

        return SimpleNamespace(
            enabled_toolsets=list(enabled),
            api_key="k", base_url="u", provider="p", api_mode="chat_completions",
            model="m", platform="cli", providers_allowed=None,
            providers_ignored=None, providers_order=None, provider_sort=None,
            _session_db=None, _delegate_depth=0, _active_children=[],
            _active_children_lock=threading.Lock(), _print_fn=None,
            tool_progress_callback=None, thinking_callback=None,
            _delegate_spinner=None, _memory_manager=None, session_id="s",
            _current_turn_id="", session_estimated_cost_usd=0.0,
            valid_tool_names=[],
        )

    def test_dropped_profile_toolsets_recorded(self):
        from tools.delegate_tool import _build_child_agent

        with patch("run_agent.AIAgent", return_value=MagicMock()):
            # Parent lacks 'web'; profile wants web+file → web is dropped.
            child = _build_child_agent(
                task_index=0, goal="g", context=None,
                toolsets=["web", "file"], model="m", max_iterations=3,
                task_count=1, parent_agent=self._parent(["file", "terminal"]),
                profile_soul="persona", profile_name="reader",
            )
        self.assertEqual(
            getattr(child, "_delegate_profile_dropped_toolsets"), ["web"]
        )

    def test_no_drop_when_parent_has_all_profile_tools(self):
        from tools.delegate_tool import _build_child_agent

        with patch("run_agent.AIAgent", return_value=MagicMock()):
            child = _build_child_agent(
                task_index=0, goal="g", context=None,
                toolsets=["web", "file"], model="m", max_iterations=3,
                task_count=1,
                parent_agent=self._parent(["file", "web", "terminal"]),
                profile_soul="persona", profile_name="reader",
            )
        self.assertEqual(
            getattr(child, "_delegate_profile_dropped_toolsets"), []
        )

    def test_blanket_stripped_toolsets_not_reported_as_dropped(self):
        # code_execution is stripped from EVERY subagent, not because the parent
        # lacks it. A profile enabling it must NOT be reported as a parent-
        # privilege drop — only genuinely unavailable tools (here 'web') should.
        from tools.delegate_tool import _build_child_agent

        with patch("run_agent.AIAgent", return_value=MagicMock()):
            child = _build_child_agent(
                task_index=0, goal="g", context=None,
                toolsets=["file", "code_execution", "web"], model="m",
                max_iterations=3, task_count=1,
                parent_agent=self._parent(["file", "code_execution"]),
                profile_soul="persona", profile_name="reader",
            )
        self.assertEqual(
            getattr(child, "_delegate_profile_dropped_toolsets"), ["web"]
        )

    def test_no_drop_field_for_non_profile_child(self):
        # Ordinary (non-profile) subagents never get a dropped-toolset list.
        from tools.delegate_tool import _build_child_agent

        with patch("run_agent.AIAgent", return_value=MagicMock()):
            child = _build_child_agent(
                task_index=0, goal="g", context=None,
                toolsets=["web"], model="m", max_iterations=3, task_count=1,
                parent_agent=self._parent(["file"]),
            )
        self.assertEqual(
            getattr(child, "_delegate_profile_dropped_toolsets"), []
        )


class TestProfileMemoryWiring(unittest.TestCase):
    """A profile-backed child loads the target profile's memory: it is built
    with skip_memory=False and profile_home pointing at the profile dir.
    Ordinary subagents stay memory-less (skip_memory=True, profile_home=None).
    """

    def _parent(self):
        from types import SimpleNamespace

        return SimpleNamespace(
            enabled_toolsets=["file", "web"],
            api_key="k", base_url="u", provider="p", api_mode="chat_completions",
            model="m", platform="cli", providers_allowed=None,
            providers_ignored=None, providers_order=None, provider_sort=None,
            _session_db=None, _delegate_depth=0, _active_children=[],
            _active_children_lock=threading.Lock(), _print_fn=None,
            tool_progress_callback=None, thinking_callback=None,
            _delegate_spinner=None, _memory_manager=None, session_id="s",
            _current_turn_id="", session_estimated_cost_usd=0.0,
            valid_tool_names=[],
        )

    def test_profile_child_loads_profile_memory(self):
        from tools.delegate_tool import _build_child_agent

        with patch("run_agent.AIAgent", return_value=MagicMock()) as MA:
            _build_child_agent(
                task_index=0, goal="g", context=None, toolsets=["file"],
                model="m", max_iterations=3, task_count=1,
                parent_agent=self._parent(), profile_soul="persona",
                profile_name="reader", profile_home="/home/x/.hermes/profiles/reader",
                override_api_key="profile-key",
            )
        kw = MA.call_args.kwargs
        self.assertFalse(kw["skip_memory"])
        self.assertEqual(kw["profile_home"], "/home/x/.hermes/profiles/reader")

    def test_ordinary_child_stays_memoryless(self):
        from tools.delegate_tool import _build_child_agent

        with patch("run_agent.AIAgent", return_value=MagicMock()) as MA:
            _build_child_agent(
                task_index=0, goal="g", context=None, toolsets=["file"],
                model="m", max_iterations=3, task_count=1,
                parent_agent=self._parent(),
            )
        kw = MA.call_args.kwargs
        self.assertTrue(kw["skip_memory"])
        self.assertIsNone(kw["profile_home"])

    def test_bundle_carries_profile_home(self):
        # _resolve_profile_bundle must expose the profile dir as profile_home.
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            prof = Path(td)
            with patch(
                "hermes_cli.profiles.profile_exists", return_value=True
            ), patch(
                "hermes_cli.profiles.get_profile_dir", return_value=prof
            ), patch(
                "hermes_cli.config.load_config",
                return_value={"model": {"default": "m", "provider": "p"}},
            ), patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                return_value={"provider": "p", "base_url": "u", "api_key": "k",
                              "api_mode": "chat_completions"},
            ), patch(
                "hermes_cli.tools_config._get_platform_tools", return_value=set()
            ):
                bundle = _resolve_profile_bundle("reader")
        self.assertEqual(bundle["profile_home"], str(prof))


class TestAgentDispatchForwardsProfile(unittest.TestCase):
    """Guard the second invocation path: the agent loop dispatches delegate_task
    via AIAgent._dispatch_delegate_task (run_agent.py), NOT the registry handler.
    That method enumerates every forwarded arg, so a new schema field silently
    breaks unless it's added there too. This regression test fails if `profile`
    (or parent_agent) stops being forwarded. See issue #41889 follow-up.
    """

    def test_dispatch_delegate_task_forwards_profile(self):
        import run_agent

        captured = {}

        def fake_delegate_task(**kwargs):
            captured.update(kwargs)
            return "{}"

        with patch("tools.delegate_tool.delegate_task", fake_delegate_task):
            # Call unbound with a throwaway `self`; the method only uses self as
            # parent_agent and imports delegate_task lazily inside.
            run_agent.AIAgent._dispatch_delegate_task(
                object(), {"profile": "reader", "goal": "g"}
            )

        self.assertEqual(captured.get("profile"), "reader")
        self.assertIn("parent_agent", captured)

    def test_dispatch_delegate_task_forwards_profile_memory(self):
        # The write-back opt-in must reach delegate_task through the live agent
        # dispatch path too, not just the registry handler.
        import run_agent

        captured = {}

        def fake_delegate_task(**kwargs):
            captured.update(kwargs)
            return "{}"

        with patch("tools.delegate_tool.delegate_task", fake_delegate_task):
            run_agent.AIAgent._dispatch_delegate_task(
                object(), {"profile": "reader", "profile_memory": "write", "goal": "g"}
            )

        self.assertEqual(captured.get("profile_memory"), "write")


class TestCoerceProfileMemory(unittest.TestCase):
    """`profile_memory` normalisation: enum + synonyms + bools, with an
    unset/unknown sentinel so resolution can fall through to the config default.
    """

    def test_write_synonyms(self):
        for v in ("write", "Write", "writeback", "rw", "readwrite", "true", True, "1"):
            self.assertIs(_coerce_profile_memory(v), True, v)

    def test_read_synonyms(self):
        for v in ("read", "READ", "readonly", "ro", "false", False, "0"):
            self.assertIs(_coerce_profile_memory(v), False, v)

    def test_unset_and_unknown_return_none(self):
        # None (unset) and unrecognised strings both fall through to the default.
        self.assertIsNone(_coerce_profile_memory(None))
        self.assertIsNone(_coerce_profile_memory(""))
        self.assertIsNone(_coerce_profile_memory("banana"))


class TestProfileMemoryWriteback(unittest.TestCase):
    """Phase-2 write-back wiring in _build_child_agent: the opt-in flips the
    child out of read-only AND grants the otherwise-stripped `memory` tool so
    the writable, profile-bound store is actually reachable. Read-only stays
    the default, and ordinary (non-profile) subagents are never affected.
    """

    def _parent(self, enabled=("file", "web", "memory")):
        from types import SimpleNamespace

        return SimpleNamespace(
            enabled_toolsets=list(enabled),
            api_key="k", base_url="u", provider="p", api_mode="chat_completions",
            model="m", platform="cli", providers_allowed=None,
            providers_ignored=None, providers_order=None, provider_sort=None,
            _session_db=None, _delegate_depth=0, _active_children=[],
            _active_children_lock=threading.Lock(), _print_fn=None,
            tool_progress_callback=None, thinking_callback=None,
            _delegate_spinner=None, _memory_manager=None, session_id="s",
            _current_turn_id="", session_estimated_cost_usd=0.0,
            valid_tool_names=[],
        )

    def _build(self, **over):
        from tools.delegate_tool import _build_child_agent

        kwargs = dict(
            task_index=0, goal="g", context=None, toolsets=["file"],
            model="m", max_iterations=3, task_count=1,
            parent_agent=self._parent(), profile_soul="persona",
            profile_name="reader", profile_home="/h/.hermes/profiles/reader",
            override_api_key="profile-key",
        )
        kwargs.update(over)
        with patch("run_agent.AIAgent", return_value=MagicMock()) as MA:
            child = _build_child_agent(**kwargs)
        return MA.call_args.kwargs, child

    def test_writeback_disables_readonly_and_grants_memory_tool(self):
        kw, child = self._build(profile_memory_writeback=True)
        self.assertFalse(kw["profile_memory_readonly"])
        self.assertIn("memory", kw["enabled_toolsets"])
        self.assertTrue(getattr(child, "_delegate_profile_writeback"))

    def test_default_is_readonly_without_memory_tool(self):
        kw, child = self._build()  # writeback defaults to False
        self.assertTrue(kw["profile_memory_readonly"])
        self.assertNotIn("memory", kw["enabled_toolsets"])
        self.assertFalse(getattr(child, "_delegate_profile_writeback"))

    def test_writeback_ignored_without_profile(self):
        # No profile_home → write-back is meaningless: never readonly-flips a
        # memoryless child, never grants the memory tool, stays not-writeback.
        from tools.delegate_tool import _build_child_agent

        with patch("run_agent.AIAgent", return_value=MagicMock()) as MA:
            child = _build_child_agent(
                task_index=0, goal="g", context=None, toolsets=["file"],
                model="m", max_iterations=3, task_count=1,
                parent_agent=self._parent(), profile_memory_writeback=True,
            )
        kw = MA.call_args.kwargs
        self.assertFalse(kw["profile_memory_readonly"])  # readonly only with profile_home
        self.assertNotIn("memory", kw["enabled_toolsets"])
        self.assertFalse(getattr(child, "_delegate_profile_writeback"))


class TestDelegateTaskWritebackResolution(unittest.TestCase):
    """delegate_task resolves the write-back opt-in with the documented
    precedence (per-task > top-level > config default) and forwards it to
    _build_child_agent.
    """

    def setUp(self):
        self.parent = _make_mock_parent()
        self._bundle = {
            "name": "reader", "soul": "Reader", "model": "m", "provider": "p",
            "base_url": "u", "api_key": "k", "api_mode": "chat_completions",
            "toolsets": None,
        }

    def _run(self, **dt):
        with patch("tools.delegate_tool._run_single_child") as mrun, patch(
            "tools.delegate_tool._build_child_agent"
        ) as mbuild, patch(
            "tools.delegate_tool._resolve_profile_bundle", return_value=self._bundle
        ), patch(
            "tools.delegate_tool._get_profile_memory_writeback", return_value=False
        ):
            fake_child = MagicMock()
            fake_child.model = "m"
            mbuild.return_value = fake_child
            mrun.return_value = {"task_index": 0, "status": "completed", "summary": "ok"}
            delegate_task(parent_agent=self.parent, **dt)
        return mbuild

    def test_top_level_write_applies_to_batch(self):
        mbuild = self._run(
            profile="reader", profile_memory="write",
            tasks=[{"goal": "a"}, {"goal": "b"}],
        )
        for call in mbuild.call_args_list:
            self.assertTrue(call.kwargs["profile_memory_writeback"])

    def test_per_task_overrides_top_level(self):
        mbuild = self._run(
            profile="reader", profile_memory="write",
            tasks=[{"goal": "a"}, {"goal": "b", "profile_memory": "read"}],
        )
        calls = mbuild.call_args_list
        self.assertTrue(calls[0].kwargs["profile_memory_writeback"])
        self.assertFalse(calls[1].kwargs["profile_memory_writeback"])

    def test_unset_falls_back_to_config_default(self):
        # No profile_memory given → uses _get_profile_memory_writeback (False here).
        mbuild = self._run(goal="g", profile="reader")
        self.assertFalse(mbuild.call_args.kwargs["profile_memory_writeback"])

    def test_unset_honors_config_default_true(self):
        with patch("tools.delegate_tool._run_single_child") as mrun, patch(
            "tools.delegate_tool._build_child_agent"
        ) as mbuild, patch(
            "tools.delegate_tool._resolve_profile_bundle", return_value=self._bundle
        ), patch(
            "tools.delegate_tool._get_profile_memory_writeback", return_value=True
        ):
            fake_child = MagicMock()
            fake_child.model = "m"
            mbuild.return_value = fake_child
            mrun.return_value = {"task_index": 0, "status": "completed", "summary": "ok"}
            delegate_task(goal="g", profile="reader", parent_agent=self.parent)
        self.assertTrue(mbuild.call_args.kwargs["profile_memory_writeback"])


class TestProfileBoundaryHardening(unittest.TestCase):
    """Profile-boundary semantics raised in PR #48644 review (NorethSea):

    - write-back is a privilege the parent can only delegate if it holds it;
    - a profile-backed child must not silently fall back to the parent's runtime;
    - interrupted/errored batch children must keep their profile identity.
    """

    def _parent(self, enabled, fallback=None):
        from types import SimpleNamespace

        return SimpleNamespace(
            enabled_toolsets=list(enabled),
            api_key="k", base_url="u", provider="p", api_mode="chat_completions",
            model="m", platform="cli", providers_allowed=None,
            providers_ignored=None, providers_order=None, provider_sort=None,
            _session_db=None, _delegate_depth=0, _active_children=[],
            _active_children_lock=threading.Lock(), _print_fn=None,
            tool_progress_callback=None, thinking_callback=None,
            _delegate_spinner=None, _memory_manager=None, session_id="s",
            _current_turn_id="", session_estimated_cost_usd=0.0,
            valid_tool_names=[], _fallback_chain=fallback,
        )

    # --- point 2: write-back gated on the parent's own memory capability ---

    def test_writeback_granted_when_parent_has_memory(self):
        from tools.delegate_tool import _build_child_agent

        with patch("run_agent.AIAgent", return_value=MagicMock()) as MA:
            child = _build_child_agent(
                task_index=0, goal="g", context=None, toolsets=["file"],
                model="m", max_iterations=3, task_count=1,
                parent_agent=self._parent(["file", "memory"]),
                profile_soul="persona", profile_name="reader",
                profile_home="/h/p/reader", profile_memory_writeback=True,
                override_api_key="profile-key",
            )
        self.assertTrue(getattr(child, "_delegate_profile_writeback"))
        self.assertIn("memory", MA.call_args.kwargs["enabled_toolsets"])

    def test_writeback_downgraded_when_parent_lacks_memory(self):
        # Parent has no 'memory' toolset → it cannot delegate a memory write, so
        # profile_memory='write' is downgraded to read-only (no amplification).
        from tools.delegate_tool import _build_child_agent

        with patch("run_agent.AIAgent", return_value=MagicMock()) as MA:
            child = _build_child_agent(
                task_index=0, goal="g", context=None, toolsets=["file"],
                model="m", max_iterations=3, task_count=1,
                parent_agent=self._parent(["file", "web"]),
                profile_soul="persona", profile_name="reader",
                profile_home="/h/p/reader", profile_memory_writeback=True,
                override_api_key="profile-key",
            )
        self.assertFalse(getattr(child, "_delegate_profile_writeback"))
        self.assertNotIn("memory", MA.call_args.kwargs["enabled_toolsets"])

    # --- point 3: profile child does not inherit the parent's fallback chain ---

    def test_profile_child_gets_no_parent_fallback(self):
        from tools.delegate_tool import _build_child_agent

        with patch("run_agent.AIAgent", return_value=MagicMock()) as MA:
            _build_child_agent(
                task_index=0, goal="g", context=None, toolsets=["file"],
                model="m", max_iterations=3, task_count=1,
                parent_agent=self._parent(["file"], fallback=["fb/model"]),
                profile_soul="persona", profile_name="reader",
                profile_home="/h/p/reader", override_api_key="profile-key",
            )
        self.assertIsNone(MA.call_args.kwargs["fallback_model"])

    def test_ordinary_child_still_inherits_parent_fallback(self):
        from tools.delegate_tool import _build_child_agent

        with patch("run_agent.AIAgent", return_value=MagicMock()) as MA:
            _build_child_agent(
                task_index=0, goal="g", context=None, toolsets=["file"],
                model="m", max_iterations=3, task_count=1,
                parent_agent=self._parent(["file"], fallback=["fb/model"]),
            )
        self.assertEqual(MA.call_args.kwargs["fallback_model"], ["fb/model"])

    # --- point 1: profile child with no key of its own fails closed -----------

    def test_profile_child_without_key_fails_closed(self):
        # A profile-backed child whose profile resolved NO runtime secret must
        # NOT inherit the parent's key (billing/audit misattribution). It raises.
        from tools.delegate_tool import _build_child_agent

        with patch("run_agent.AIAgent", return_value=MagicMock()):
            with self.assertRaises(ValueError) as ctx:
                _build_child_agent(
                    task_index=0, goal="g", context=None, toolsets=["file"],
                    model="m", max_iterations=3, task_count=1,
                    parent_agent=self._parent(["file"]),
                    profile_soul="persona", profile_name="reader",
                    profile_home="/h/p/reader",  # no override_api_key
                )
        self.assertIn("reader", str(ctx.exception))
        self.assertIn("no runtime secret", str(ctx.exception))

    def test_failclosed_surfaces_as_tool_error_not_crash(self):
        # Through the real delegate_task path, a keyless profile must return a
        # clean tool error (not raise out of the turn). Mirrors the bundle-error
        # contract in test_invalid_profile_returns_tool_error.
        import json
        from tools.delegate_tool import delegate_task

        bundle = {
            "name": "reader", "soul": "persona", "model": "m", "provider": "p",
            "base_url": "u", "api_key": None, "api_mode": "chat_completions",
            "toolsets": ["file"], "profile_home": "/h/p/reader",
        }
        with patch(
            "tools.delegate_tool._resolve_profile_bundle", return_value=bundle
        ), patch("run_agent.AIAgent", return_value=MagicMock()):
            out = delegate_task(
                goal="g", profile="reader",
                parent_agent=self._parent(["file"]),
            )
        data = json.loads(out)
        self.assertIn("error", data)
        self.assertIn("no runtime secret", data["error"])

    def test_ordinary_child_without_override_key_still_inherits(self):
        # The fail-closed rule is scoped to PROFILE children only: an ordinary
        # (non-profile) subagent still inherits the parent's key as before.
        from tools.delegate_tool import _build_child_agent

        with patch("run_agent.AIAgent", return_value=MagicMock()) as MA:
            _build_child_agent(
                task_index=0, goal="g", context=None, toolsets=["file"],
                model="m", max_iterations=3, task_count=1,
                parent_agent=self._parent(["file"]),  # parent api_key="k"
            )
        self.assertEqual(MA.call_args.kwargs["api_key"], "k")

    # --- point 5: fabricated interrupted/errored entries keep profile identity --

    def test_profile_fields_from_child_reads_identity(self):
        from types import SimpleNamespace
        from tools.delegate_tool import _profile_fields_from_child

        child = SimpleNamespace(
            _delegate_profile="reader",
            _delegate_profile_writeback=False,
            _delegate_profile_dropped_toolsets=["web"],
        )
        fields = _profile_fields_from_child(child)
        self.assertEqual(fields["profile"], "reader")
        self.assertEqual(fields["profile_memory"], "read")
        self.assertEqual(fields["profile_toolsets_dropped"], ["web"])

    def test_profile_fields_from_child_is_mock_safe(self):
        # An ordinary (non-profile) or mock child must not leak MagicMock attrs
        # into the JSON entry: profile None, profile_memory None, no dropped key.
        from tools.delegate_tool import _profile_fields_from_child

        fields = _profile_fields_from_child(MagicMock())
        self.assertIsNone(fields["profile"])
        self.assertIsNone(fields["profile_memory"])
        self.assertNotIn("profile_toolsets_dropped", fields)


if __name__ == "__main__":
    unittest.main()
