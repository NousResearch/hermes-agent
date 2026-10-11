"""Operator-controlled delegation route and credential-boundary regressions."""

import json
import unittest
from unittest.mock import MagicMock, patch

from hermes_cli.config_defaults import DEFAULT_CONFIG
from tests.tools.test_delegate import _make_mock_parent
from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    _build_child_agent,
    _check_credential_authority,
    _configured_delegation_routes,
    _normalize_endpoint_authority,
    _resolve_delegation_credentials,
    _resolve_delegation_route_config,
    delegate_task,
)


class TestDelegationDefaultToolsets(unittest.TestCase):
    def _build(self, config, parent_enabled, *, role="leaf"):
        parent = _make_mock_parent()
        parent.enabled_toolsets = parent_enabled
        with (
            patch("tools.delegate_tool._load_config", return_value=config),
            patch("run_agent.AIAgent") as MockAgent,
        ):
            MockAgent.return_value = MagicMock()
            _build_child_agent(
                task_index=0,
                goal="test",
                context=None,
                toolsets=None,
                model="test-model",
                max_iterations=5,
                parent_agent=parent,
                task_count=1,
                role=role,
            )
        return MockAgent.call_args.kwargs

    def test_absent_config_preserves_parent_inheritance(self):
        kwargs = self._build({}, ["terminal", "file", "web"])
        self.assertEqual(kwargs["enabled_toolsets"], ["terminal", "file", "web"])

    def test_operator_route_and_toolset_defaults_are_registered(self):
        delegation = DEFAULT_CONFIG["delegation"]
        self.assertIsNone(delegation["default_toolsets"])
        self.assertEqual(delegation["routes"], {})
        self.assertIs(delegation["inherit_mcp_toolsets"], True)

    def test_configured_list_narrows_parent_capabilities(self):
        kwargs = self._build(
            {"default_toolsets": ["file"]},
            ["terminal", "file", "web"],
        )
        self.assertEqual(kwargs["enabled_toolsets"], ["file"])

    def test_configured_list_is_a_ceiling_over_inherited_mcp_toolsets(self):
        kwargs = self._build(
            {"default_toolsets": ["file"], "inherit_mcp_toolsets": True},
            ["terminal", "file", "mcp-qdrant"],
        )
        self.assertEqual(kwargs["enabled_toolsets"], ["file"])

    def test_configured_empty_list_enables_no_child_toolsets(self):
        kwargs = self._build(
            {"default_toolsets": []},
            ["terminal", "file", "web", "mcp-qdrant"],
        )
        self.assertEqual(kwargs["enabled_toolsets"], [])

    def test_orchestrator_without_ceiling_retains_delegation(self):
        kwargs = self._build(
            {"max_spawn_depth": 2, "orchestrator_enabled": True},
            ["terminal", "file"],
            role="orchestrator",
        )
        self.assertIn("delegation", kwargs["enabled_toolsets"])
        self.assertNotIn("delegation", kwargs["disabled_toolsets"])

    def test_orchestrator_empty_ceiling_blocks_delegation(self):
        kwargs = self._build(
            {
                "default_toolsets": [],
                "max_spawn_depth": 2,
                "orchestrator_enabled": True,
            },
            ["terminal", "file", "delegation"],
            role="orchestrator",
        )
        self.assertEqual(kwargs["enabled_toolsets"], [])
        self.assertIn("delegation", kwargs["disabled_toolsets"])

    def test_orchestrator_ceiling_can_include_delegation(self):
        kwargs = self._build(
            {
                "default_toolsets": ["file", "delegation"],
                "max_spawn_depth": 2,
                "orchestrator_enabled": True,
            },
            ["terminal", "file", "delegation"],
            role="orchestrator",
        )
        self.assertEqual(kwargs["enabled_toolsets"], ["file", "delegation"])
        self.assertNotIn("delegation", kwargs["disabled_toolsets"])

    def test_configured_list_cannot_grant_unknown_or_blocked_toolsets(self):
        kwargs = self._build(
            {"default_toolsets": ["file", "not-a-toolset", "delegation", "memory"]},
            ["terminal", "file", "web"],
        )
        self.assertEqual(kwargs["enabled_toolsets"], ["file"])
        self.assertIn("delegation", kwargs["disabled_toolsets"])
        self.assertIn("memory", kwargs["disabled_toolsets"])

    def test_invalid_config_type_preserves_parent_inheritance(self):
        kwargs = self._build(
            {"default_toolsets": "file"},
            ["terminal", "file", "web"],
        )
        self.assertEqual(kwargs["enabled_toolsets"], ["terminal", "file", "web"])


class TestDelegationRouteSchema(unittest.TestCase):
    def test_routes_are_absent_when_operator_configures_none(self):
        from tools.delegate_tool import _build_dynamic_schema_overrides

        with patch("tools.delegate_tool._load_config", return_value={}):
            props = _build_dynamic_schema_overrides()["parameters"]["properties"]
        self.assertNotIn("route", props)
        self.assertNotIn("route", props["tasks"]["items"]["properties"])

    def test_routes_expose_only_operator_configured_names(self):
        from tools.delegate_tool import _build_dynamic_schema_overrides

        cfg = {
            "routes": {
                "local": {"model": "local-model"},
                "hosted-fast": {"model": "hosted-model", "provider": "custom"},
                "bad route": {"model": "must-not-appear"},
                "not-a-map": "bad",
            }
        }
        with patch("tools.delegate_tool._load_config", return_value=cfg):
            props = _build_dynamic_schema_overrides()["parameters"]["properties"]
        self.assertEqual(props["route"]["enum"], ["hosted-fast", "local"])
        self.assertEqual(
            props["tasks"]["items"]["properties"]["route"]["enum"],
            ["hosted-fast", "local"],
        )
        self.assertNotIn("provider", props)
        self.assertNotIn("model", props)
        self.assertNotIn("base_url", props)
        self.assertNotIn("api_key", props)
        self.assertNotIn("api_mode", props)

        # Dynamic schema assembly must not mutate the import-time schema or
        # leak route names into sessions/configurations that do not define them.
        with patch("tools.delegate_tool._load_config", return_value={}):
            absent_props = _build_dynamic_schema_overrides()["parameters"]["properties"]
        self.assertNotIn("route", absent_props)
        self.assertNotIn(
            "route", absent_props["tasks"]["items"]["properties"]
        )


class TestConfiguredDelegationRoutes(unittest.TestCase):
    def test_empty_routes_config(self):
        self.assertEqual(_configured_delegation_routes({}), {})

    def test_non_dict_routes_ignored(self):
        self.assertEqual(_configured_delegation_routes({"routes": "bad"}), {})

    def test_allowlisted_fields_only(self):
        routes = _configured_delegation_routes({
            "routes": {
                "myroute": {
                    "model": "m",
                    "provider": "p",
                    "base_url": "http://x/v1",
                    "api_key": "k",
                    "api_mode": "chat_completions",
                    "request_overrides": {"foo": 1},  # route-owned transport setting
                    "max_iterations": 99,  # not allowlisted
                }
            }
        })
        self.assertEqual(set(routes["myroute"]), {"model", "provider", "base_url", "api_key", "api_mode", "request_overrides"})

    def test_invalid_names_skipped(self):
        routes = _configured_delegation_routes({
            "routes": {
                "valid-route": {"model": "m"},
                "bad route": {"model": "m"},  # space not allowed
                "": {"model": "m"},           # empty not allowed
                "!bad": {"model": "m"},       # special char
            }
        })
        self.assertEqual(list(routes), ["valid-route"])

    def test_non_dict_entry_skipped(self):
        routes = _configured_delegation_routes({
            "routes": {
                "ok": {"model": "m"},
                "bad": "not-a-dict",
            }
        })
        self.assertIn("ok", routes)
        self.assertNotIn("bad", routes)


class TestDelegationRoutes(unittest.TestCase):
    @staticmethod
    def _creds_for(cfg, _parent):
        return {
            "model": cfg.get("model"),
            "provider": cfg.get("provider"),
            "base_url": None,
            "api_key": "test-route-key",
            "api_mode": None,
            "request_overrides": None,
            "max_output_tokens": None,
            "command": None,
            "args": [],
        }

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_batch_routes_each_child_through_named_operator_preset(
        self, mock_creds, mock_cfg
    ):
        cfg = {
            "model": "default-model",
            "provider": "default-provider",
            "max_iterations": 17,
            "default_toolsets": ["file"],
            "routes": {
                "local": {"model": "local-model", "provider": "local-provider"},
                "hosted": {"model": "hosted-model", "provider": "hosted-provider"},
            },
        }
        mock_cfg.return_value = cfg
        mock_creds.side_effect = self._creds_for
        parent = _make_mock_parent()

        with patch("run_agent.AIAgent") as MockAgent:
            children = [MagicMock(), MagicMock()]
            for child in children:
                child.run_conversation.return_value = {
                    "final_response": "done",
                    "completed": True,
                    "api_calls": 1,
                }
            MockAgent.side_effect = children
            delegate_task(
                tasks=[
                    {"goal": "local work", "route": "local"},
                    {"goal": "hosted work", "route": "hosted"},
                ],
                parent_agent=parent,
            )

        kwargs = [call.kwargs for call in MockAgent.call_args_list]
        self.assertEqual([item["model"] for item in kwargs], ["local-model", "hosted-model"])
        self.assertEqual(
            [item["provider"] for item in kwargs],
            ["local-provider", "hosted-provider"],
        )
        resolved_cfgs = [call.args[0] for call in mock_creds.call_args_list]
        self.assertNotIn("routes", resolved_cfgs[0])
        self.assertEqual(resolved_cfgs[0]["model"], "local-model")
        self.assertEqual(resolved_cfgs[1]["model"], "hosted-model")
        self.assertEqual(resolved_cfgs[0]["max_iterations"], 17)
        self.assertEqual(resolved_cfgs[0]["default_toolsets"], ["file"])

    def test_top_level_route_is_batch_fallback_only(self):
        cfg = {
            "routes": {
                "local": {"model": "local-model", "provider": "local-provider"},
                "hosted": {"model": "hosted-model", "provider": "hosted-provider"},
            }
        }
        parent = _make_mock_parent()
        with (
            patch("tools.delegate_tool._load_config", return_value=cfg),
            patch(
                "tools.delegate_tool._resolve_delegation_credentials",
                side_effect=self._creds_for,
            ),
            patch("run_agent.AIAgent") as MockAgent,
        ):
            children = [MagicMock(), MagicMock()]
            for child in children:
                child.run_conversation.return_value = {
                    "final_response": "done", "completed": True, "api_calls": 1
                }
            MockAgent.side_effect = children
            delegate_task(
                tasks=[
                    {"goal": "inherits fallback"},
                    {"goal": "explicit local", "route": "local"},
                ],
                route="hosted",
                parent_agent=parent,
            )
        kwargs = [call.kwargs for call in MockAgent.call_args_list]
        self.assertEqual(
            [item["model"] for item in kwargs],
            ["hosted-model", "local-model"],
        )

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_unknown_route_fails_closed_before_credential_resolution(
        self, mock_creds, mock_cfg
    ):
        mock_cfg.return_value = {
            "routes": {"local": {"model": "local-model"}}
        }
        result = json.loads(
            delegate_task(
                goal="work",
                route="attacker/provider-model",
                parent_agent=_make_mock_parent(),
            )
        )
        self.assertIn("Unknown delegation route", result["error"])
        self.assertIn("local", result["error"])
        mock_creds.assert_not_called()

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_later_unknown_route_fails_before_any_credential_resolution(
        self, mock_creds, mock_cfg
    ):
        mock_cfg.return_value = {
            "routes": {"local": {"model": "local-model"}}
        }
        result = json.loads(
            delegate_task(
                tasks=[
                    {"goal": "valid first", "route": "local"},
                    {"goal": "invalid second", "route": "attacker/provider-model"},
                ],
                parent_agent=_make_mock_parent(),
            )
        )
        self.assertIn("Unknown delegation route", result["error"])
        mock_creds.assert_not_called()

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_absent_route_preserves_legacy_top_level_config(
        self, mock_creds, mock_cfg
    ):
        cfg = {"model": "legacy-model", "provider": "legacy-provider"}
        mock_cfg.return_value = cfg
        mock_creds.side_effect = self._creds_for
        parent = _make_mock_parent()
        with patch("run_agent.AIAgent") as MockAgent:
            child = MagicMock()
            child.run_conversation.return_value = {
                "final_response": "done", "completed": True, "api_calls": 1
            }
            MockAgent.return_value = child
            delegate_task(goal="legacy work", parent_agent=parent)
        self.assertEqual(MockAgent.call_args.kwargs["model"], "legacy-model")
        self.assertEqual(MockAgent.call_args.kwargs["provider"], "legacy-provider")
        self.assertEqual(mock_creds.call_count, 1)
        self.assertIs(mock_creds.call_args.args[0], cfg)

    def test_resolve_delegation_route_config_overlays_route_fields(self):
        cfg = {
            "model": "base-model",
            "provider": "base-provider",
            "max_iterations": 100,
            "routes": {
                "fast": {"model": "fast-model", "provider": "fast-provider"},
            },
        }
        effective = _resolve_delegation_route_config(cfg, "fast")
        self.assertEqual(effective["model"], "fast-model")
        self.assertEqual(effective["provider"], "fast-provider")
        self.assertEqual(effective["max_iterations"], 100)
        self.assertNotIn("routes", effective)

    def test_named_route_does_not_reuse_legacy_key_for_changed_authority(self):
        """A top-level legacy key is not explicit authority for a named route."""
        cfg = {
            "base_url": "https://legacy.example/v1",
            "api_key": "LEGACY",
            "routes": {
                "isolated": {"model": "test-model", "base_url": "https://route.example/v1"},
            },
        }
        parent = _make_mock_parent()
        parent.base_url = "https://legacy.example/v1"
        parent.api_key = "LEGACY"

        effective = _resolve_delegation_route_config(cfg, "isolated")
        self.assertNotIn("api_key", effective)
        with self.assertRaises(ValueError) as ctx:
            _resolve_delegation_credentials(effective, parent)

        self.assertIn("authority mismatch", str(ctx.exception))

    def test_resolve_delegation_route_config_unknown_route_raises(self):
        cfg = {"routes": {"local": {"model": "m"}}}
        with self.assertRaises(ValueError) as ctx:
            _resolve_delegation_route_config(cfg, "nonexistent")
        self.assertIn("Unknown delegation route", str(ctx.exception))
        self.assertIn("local", str(ctx.exception))

    def test_resolve_delegation_route_config_none_route_returns_cfg_unchanged(self):
        cfg = {"model": "m", "routes": {"r": {"model": "r-model"}}}
        self.assertIs(_resolve_delegation_route_config(cfg, None), cfg)
        self.assertIs(_resolve_delegation_route_config(cfg, ""), cfg)
        self.assertIs(_resolve_delegation_route_config(cfg, "  "), cfg)


class TestCredentialAuthority(unittest.TestCase):
    """Tests for endpoint credential authority enforcement."""

    def test_normalize_endpoint_authority_basic(self):
        self.assertEqual(
            _normalize_endpoint_authority("https://api.example.com/v1"),
            ("https", "api.example.com", 443),
        )

    def test_normalize_endpoint_authority_strips_path(self):
        self.assertEqual(
            _normalize_endpoint_authority("http://localhost:1234/v1/chat"),
            ("http", "localhost", 1234),
        )

    def test_normalize_endpoint_authority_default_port_is_explicit(self):
        self.assertEqual(
            _normalize_endpoint_authority("https://api.example.com:443/v1"),
            ("https", "api.example.com", 443),
        )
        self.assertEqual(
            _normalize_endpoint_authority("http://api.example.com:80/v1"),
            ("http", "api.example.com", 80),
        )

    def test_normalize_endpoint_authority_non_default_port_kept(self):
        self.assertEqual(
            _normalize_endpoint_authority("https://api.example.com:8443/v1"),
            ("https", "api.example.com", 8443),
        )

    def test_bracketed_ipv6_authorities_preserve_port_boundary(self):
        self.assertNotEqual(
            _normalize_endpoint_authority("https://[2001:db8::1]:8443/v1"),
            _normalize_endpoint_authority("https://[2001:db8::1:8443]/v1"),
        )

    def test_normalize_endpoint_authority_none_returns_none(self):
        self.assertIsNone(_normalize_endpoint_authority(None))
        self.assertIsNone(_normalize_endpoint_authority(""))

    def test_same_authority_no_explicit_key_passes(self):
        """Same scheme/host/port: parent key inheritance is safe."""
        parent = _make_mock_parent()
        parent.base_url = "https://openrouter.ai/api/v1"
        err = _check_credential_authority(
            parent,
            configured_base_url="https://openrouter.ai/api/v1",
            configured_api_key=None,
        )
        self.assertIsNone(err)

    def test_different_authority_no_explicit_key_fails(self):
        """Changed endpoint without api_key must be rejected."""
        parent = _make_mock_parent()
        parent.base_url = "https://openrouter.ai/api/v1"
        err = _check_credential_authority(
            parent,
            configured_base_url="https://evil.example.com/v1",
            configured_api_key=None,
        )
        self.assertIsNotNone(err)
        self.assertIn("authority mismatch", err)

    def test_different_authority_with_explicit_key_passes(self):
        """Explicit key provided: authority restriction does not apply."""
        parent = _make_mock_parent()
        parent.base_url = "https://openrouter.ai/api/v1"
        err = _check_credential_authority(
            parent,
            configured_base_url="https://evil.example.com/v1",
            configured_api_key="sk-explicit-key",
        )
        self.assertIsNone(err)

    def test_no_base_url_skips_check(self):
        """No configured_base_url: authority check is bypassed."""
        parent = _make_mock_parent()
        err = _check_credential_authority(parent, configured_base_url=None, configured_api_key=None)
        self.assertIsNone(err)

    def test_resolve_credentials_raises_on_authority_mismatch(self):
        """_resolve_delegation_credentials raises ValueError for mismatched endpoint without key."""
        parent = _make_mock_parent()
        parent.base_url = "https://openrouter.ai/api/v1"
        cfg = {"base_url": "https://different-host.com/v1"}
        with self.assertRaises(ValueError) as ctx:
            _resolve_delegation_credentials(cfg, parent)
        self.assertIn("authority mismatch", str(ctx.exception))

    def test_resolve_credentials_same_authority_inherits_key(self):
        """Same authority without api_key resolves without error (inherits parent key)."""
        parent = _make_mock_parent()
        parent.base_url = "https://openrouter.ai/api/v1"
        cfg = {"base_url": "https://openrouter.ai/api/v1"}
        with patch("hermes_cli.runtime_provider._detect_api_mode_for_url", return_value=None):
            result = _resolve_delegation_credentials(cfg, parent)
        self.assertIsNone(result["api_key"])  # inherited from parent

    def test_live_parent_snapshot_drives_authority_and_child_credentials(self):
        """Validation and construction must ignore stale credential attributes."""
        parent = _make_mock_parent()
        parent.base_url = "https://stale.example/v1"
        parent.api_key = "STALE"
        parent._client_kwargs = {
            "base_url": "https://live.example/v1",
            "api_key": "LIVE",
        }
        cfg = {"base_url": "https://live.example/v1"}

        with patch(
            "hermes_cli.runtime_provider._detect_api_mode_for_url",
            return_value=None,
        ):
            resolved = _resolve_delegation_credentials(cfg, parent)

        with patch("run_agent.AIAgent") as MockAgent:
            MockAgent.return_value = MagicMock()
            _build_child_agent(
                task_index=0,
                goal="Use the live parent route",
                context=None,
                toolsets=None,
                model=None,
                max_iterations=10,
                parent_agent=parent,
                task_count=1,
                override_provider=resolved["provider"],
                override_base_url=resolved["base_url"],
                override_api_key=resolved["api_key"],
                override_api_mode=resolved["api_mode"],
            )

        self.assertEqual(MockAgent.call_args.kwargs["base_url"], "https://live.example/v1")
        self.assertEqual(MockAgent.call_args.kwargs["api_key"], "LIVE")

    def test_stale_surface_endpoint_cannot_receive_live_parent_key(self):
        """A stale matching surface URL cannot bypass the live authority check."""
        parent = _make_mock_parent()
        parent.base_url = "https://stale.example/v1"
        parent.api_key = "STALE"
        parent._client_kwargs = {
            "base_url": "https://live.example/v1",
            "api_key": "LIVE",
        }

        err = _check_credential_authority(
            parent,
            configured_base_url="https://stale.example/v1",
            configured_api_key=None,
        )

        self.assertIsNotNone(err)
        self.assertIn("authority mismatch", err)

    def test_resolve_credentials_different_authority_with_key_succeeds(self):
        """Changed endpoint with explicit api_key must succeed."""
        parent = _make_mock_parent()
        parent.base_url = "https://openrouter.ai/api/v1"
        cfg = {"base_url": "https://other.provider.com/v1", "api_key": "explicit-key"}
        with patch("hermes_cli.runtime_provider._detect_api_mode_for_url", return_value=None):
            result = _resolve_delegation_credentials(cfg, parent)
        self.assertEqual(result["api_key"], "explicit-key")


class TestRouteSchemaCache(unittest.TestCase):
    """Cache invalidation for managed-overlay and same-(mtime_ns,size) rewrites."""

    def test_managed_content_signature_included_in_tool_defs_cache_key(self):
        """A managed-config identity change invalidates the schema cache."""
        import model_tools

        model_tools._tool_defs_cache.clear()
        signatures = [
            ("/managed-a/config.yaml", 1000, 100, "a" * 64),
            ("/managed-b/config.yaml", 1000, 100, "b" * 64),
        ]
        with (
            patch("model_tools._tool_defs_config_fingerprints") as fingerprints,
            patch("model_tools.check_fn_cache_scope", return_value="test-scope"),
            patch("model_tools._compute_tool_definitions", return_value=[]) as compute,
        ):
            fingerprints.side_effect = [
                (None, signatures[0]),
                (None, signatures[0]),
                (None, signatures[1]),
                (None, signatures[1]),
            ]
            model_tools.get_tool_definitions(quiet_mode=True)
            model_tools.get_tool_definitions(quiet_mode=True)

        self.assertEqual(compute.call_count, 2)

    def test_route_schema_does_not_mutate_import_time_schema(self):
        """_build_dynamic_schema_overrides must not modify DELEGATE_TASK_SCHEMA."""
        from tools.delegate_tool import _build_dynamic_schema_overrides
        import copy as _copy

        original = _copy.deepcopy(DELEGATE_TASK_SCHEMA)
        cfg = {
            "routes": {
                "fast": {"model": "fast-model", "provider": "custom"},
            }
        }
        with patch("tools.delegate_tool._load_config", return_value=cfg):
            _build_dynamic_schema_overrides()

        # Static schema must be unchanged
        self.assertEqual(DELEGATE_TASK_SCHEMA, original)

        # Subsequent call without routes must not see leftovers
        with patch("tools.delegate_tool._load_config", return_value={}):
            props = _build_dynamic_schema_overrides()["parameters"]["properties"]
        self.assertNotIn("route", props)
        self.assertNotIn("route", props["tasks"]["items"]["properties"])

class TestDispatchRouteForwarding(unittest.TestCase):
    def test_top_level_route_forwarded(self):
        """The live model dispatch path must preserve the selected route."""
        import run_agent

        captured = {}

        def fake_delegate_task(**kwargs):
            captured.update(kwargs)
            return "{}"

        parent = _make_mock_parent(depth=0)
        with patch("tools.delegate_tool.delegate_task", fake_delegate_task):
            run_agent.AIAgent._dispatch_delegate_task(
                parent,
                {"tasks": [{"goal": "test"}], "route": "hosted"},
            )

        self.assertEqual(captured["route"], "hosted")

    def test_registry_fallback_forwards_top_level_route(self):
        """The registry fallback must match the live model dispatch path."""
        from tools.registry import registry

        captured = {}

        def fake_delegate_task(**kwargs):
            captured.update(kwargs)
            return "{}"

        parent = _make_mock_parent(depth=0)
        with patch("tools.delegate_tool.delegate_task", fake_delegate_task):
            registry.dispatch(
                "delegate_task",
                {"tasks": [{"goal": "test"}], "route": "local"},
                parent_agent=parent,
            )

        self.assertEqual(captured["route"], "local")
