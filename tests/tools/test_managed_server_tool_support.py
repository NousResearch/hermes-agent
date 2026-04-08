"""
Tests for ManagedServer / tool-parser integration.

Validates that:
1. The installed atroposlib API still matches Hermes's expectations
2. Hermes's parser registry remains compatible with ManagedServer parsing
3. HermesAgentBaseEnv wires the selected parser into ServerManager correctly

These tests verify the contract between hermes-agent's environments/ code
and atroposlib's ManagedServer. They detect API incompatibilities early.
"""

import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import atroposlib  # noqa: F401
except ImportError:
    pytest.skip("atroposlib not installed", allow_module_level=True)


class TestManagedServerAPI:
    """Test that ManagedServer's API matches what hermes-agent expects."""

    def test_managed_server_init_signature(self):
        """ManagedServer should accept tool_call_parser parameter."""
        from atroposlib.envs.server_handling.managed_server import ManagedServer

        sig = inspect.signature(ManagedServer.__init__)
        params = list(sig.parameters.keys())

        # Core params that must exist
        assert "self" in params
        assert "server" in params
        assert "tokenizer" in params
        assert "track_tree" in params

        # tool_call_parser — required for tool_call_support branch
        # If this fails, atroposlib hasn't been updated to tool_call_support
        has_tool_parser = "tool_call_parser" in params
        if not has_tool_parser:
            pytest.skip(
                "ManagedServer does not have tool_call_parser param — "
                "baseline atroposlib (pre tool_call_support branch)"
            )

    def test_server_manager_managed_server_signature(self):
        """ServerManager.managed_server() should accept tool_call_parser."""
        from atroposlib.envs.server_handling.server_manager import ServerManager

        sig = inspect.signature(ServerManager.managed_server)
        params = list(sig.parameters.keys())

        assert "self" in params
        assert "tokenizer" in params

        has_tool_parser = "tool_call_parser" in params
        if not has_tool_parser:
            pytest.skip(
                "ServerManager.managed_server() does not have tool_call_parser param — "
                "baseline atroposlib (pre tool_call_support branch)"
            )

    def test_managed_server_chat_template_kwargs(self):
        """ManagedServer should have CHAT_TEMPLATE_KWARGS for forwarding tools/thinking."""
        from atroposlib.envs.server_handling.managed_server import ManagedServer

        if not hasattr(ManagedServer, "CHAT_TEMPLATE_KWARGS"):
            pytest.skip(
                "ManagedServer does not have CHAT_TEMPLATE_KWARGS — "
                "baseline atroposlib (pre tool_call_support branch)"
            )

        kwargs = ManagedServer.CHAT_TEMPLATE_KWARGS
        assert "tools" in kwargs, "tools must be in CHAT_TEMPLATE_KWARGS"

    def test_no_get_logprobs_method(self):
        """get_logprobs should be removed in tool_call_support branch."""
        from atroposlib.envs.server_handling.managed_server import ManagedServer

        # In baseline, get_logprobs exists. In tool_call_support, it's removed.
        # We just note the state — not a hard fail either way.
        has_get_logprobs = hasattr(ManagedServer, "get_logprobs")
        if has_get_logprobs:
            pytest.skip(
                "ManagedServer still has get_logprobs — baseline atroposlib"
            )


class TestParserCompatibility:
    """Test that hermes-agent's parsers match ManagedServer's expectations."""

    def test_parser_parse_returns_correct_format(self):
        """
        ManagedServer expects parser.parse(text) -> (content, tool_calls)
        where tool_calls is a list of objects with .id, .function.name, .function.arguments
        """
        from environments.tool_call_parsers import get_parser

        parser = get_parser("hermes")
        text = '<tool_call>{"name": "terminal", "arguments": {"command": "ls"}}</tool_call>'
        content, tool_calls = parser.parse(text)

        assert tool_calls is not None
        assert len(tool_calls) == 1

        tc = tool_calls[0]
        # ManagedServer accesses these attrs directly
        assert hasattr(tc, "id")
        assert hasattr(tc, "function")
        assert hasattr(tc.function, "name")
        assert hasattr(tc.function, "arguments")

    def test_parser_no_tools_returns_none(self):
        """ManagedServer checks `if parsed_tool_calls:` — None should be falsy."""
        from environments.tool_call_parsers import get_parser

        parser = get_parser("hermes")
        content, tool_calls = parser.parse("Just text, no tools")
        assert tool_calls is None

    def test_parser_content_is_string_or_none(self):
        """ManagedServer uses `parsed_content or ""` — must be str or None."""
        from environments.tool_call_parsers import get_parser

        parser = get_parser("hermes")

        # With tool calls
        text = '<tool_call>{"name": "terminal", "arguments": {"command": "ls"}}</tool_call>'
        content, _ = parser.parse(text)
        assert content is None or isinstance(content, str)

        # Without tool calls
        content2, _ = parser.parse("Just text")
        assert isinstance(content2, str)


class TestBaseEnvCompatibility:
    """Test that hermes_base_env.py's tool-parser wiring matches the current API."""

    def test_hermes_base_env_sets_server_manager_tool_parser(self):
        """Hermes wires parser selection through ServerManager.tool_parser."""
        import ast

        base_env_path = Path(__file__).parent.parent / "environments" / "hermes_base_env.py"
        source = base_env_path.read_text()
        tree = ast.parse(source)

        found_assignment = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and target.attr == "tool_parser":
                        parent = target.value
                        if (
                            isinstance(parent, ast.Attribute)
                            and parent.attr == "server"
                            and isinstance(parent.value, ast.Name)
                            and parent.value.id == "self"
                        ):
                            found_assignment = True

        assert found_assignment, (
            "hermes_base_env.py should set self.server.tool_parser from config.tool_call_parser"
        )

    def test_hermes_base_env_uses_config_tool_call_parser(self):
        """Verify hermes_base_env uses the config field rather than a local parser instance."""
        base_env_path = Path(__file__).parent.parent / "environments" / "hermes_base_env.py"
        source = base_env_path.read_text()

        assert 'tool_call_parser: str = Field(' in source
        assert 'self.server.tool_parser = config.tool_call_parser' in source
