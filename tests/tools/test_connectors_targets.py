"""manage_connections target validation: an MCP target must OMIT 'alias' and 'to'.

A model that sent ``{"name": "https://mcp.example.com/mcp", "mcp": true, "alias": "", "to": "x"}``
was refused with a terse message and retried the same shape. Rejecting any alias/to key on an
``mcp: true`` target (empty, null or non-empty) is deliberate and stays strict; these tests pin
that the refusal tells the model how to recover. Nothing is repaired silently and nothing
touches the network, the MCP installer or OAuth.
"""

import json
import re
from unittest.mock import patch

import pytest

from tools.connectors.targets import HostedTarget, normalize_targets, validate_action
from tools.connectors.tool import manage_connections

EXAMPLE_URL = "https://mcp.example.com/mcp"


def _mcp_target(**extra):
    return {"name": "linear", "mcp": True, **extra}


def test_mcp_target_without_alias_or_to_is_accepted():
    managed, mcp, error = normalize_targets([{"name": "Linear", "mcp": True}])
    assert (managed, mcp, error) == ([], ["linear"], None)


@pytest.mark.parametrize("key", ["alias", "to"])
@pytest.mark.parametrize("value", ["", "x", None], ids=["empty", "nonempty", "null"])
def test_mcp_target_rejects_alias_or_to_even_when_empty_or_null(key, value):
    managed, mcp, error = normalize_targets([_mcp_target(**{key: value})])
    assert (managed, mcp) == ([], [])
    assert error and f"'{key}'" in error


@pytest.mark.parametrize(
    "fields",
    [{"alias": ""}, {"to": None}, {"alias": "x", "to": "y"}],
    ids=["alias", "to", "both"],
)
def test_rejection_tells_the_model_how_to_recover(fields):
    _, _, error = normalize_targets([_mcp_target(**fields)])
    for key in fields:
        assert f"'{key}'" in error
    assert "OMIT" in error
    # The error shows the valid target shape: a JSON object with exactly name + mcp:true.
    shapes = [json.loads(m) for m in re.findall(r"\{[^{}]*\}", error)]
    assert any(set(s) == {"name", "mcp"} and s["mcp"] is True for s in shapes)
    # A custom server URL is routed to the CLI, not to manage_connections.
    assert re.search(r"hermes mcp add \S+ --url \S+", error)


def test_rejection_does_not_repair_input():
    raw = [_mcp_target(alias="", to="x")]
    before = json.dumps(raw, sort_keys=True)
    managed, mcp, error = normalize_targets(raw)
    assert error and (managed, mcp) == ([], [])
    assert json.dumps(raw, sort_keys=True) == before


def test_hosted_alias_and_rename_are_preserved():
    managed, mcp, error = normalize_targets([{"name": "gmail", "alias": "work"}])
    assert (managed, mcp, error) == ([HostedTarget("gmail", "work", None)], [], None)
    assert validate_action("connect", managed, mcp) is None

    managed, mcp, error = normalize_targets([{"name": "gmail", "alias": "work", "to": "home"}])
    assert error is None
    assert managed == [HostedTarget("gmail", "work", "home")]
    assert validate_action("rename", managed, mcp) is None
    assert validate_action("connect", managed, mcp)


class _ForbiddenBackend:
    """Any attribute access means the MCP installer was reached."""

    def __getattr__(self, name):
        raise AssertionError(f"external effect reached: backend.{name}")


def test_install_handler_refuses_url_with_alias_and_to_offline():
    connectors = [{"name": EXAMPLE_URL, "mcp": True, "alias": "", "to": "x"}]
    with patch("tools.connectors.tool.run_managed_action", side_effect=AssertionError("managed action reached")), \
         patch("tools.connectors.mcp.open_runner", side_effect=AssertionError("runner opened")):
        result = json.loads(manage_connections(
            {"action": "install", "connectors": connectors},
            mcp_backend=_ForbiddenBackend(),
            connectors_available=lambda: False,
        ))
    error = result["error"]
    assert "OMIT" in error
    assert re.search(r"hermes mcp add \S+ --url \S+", error)
