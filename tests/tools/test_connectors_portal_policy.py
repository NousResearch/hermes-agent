"""Connector catalog and policy wire and composition contracts."""

from types import SimpleNamespace

import pytest

from tools.connectors.portal.policy import (
    InvalidMemberPolicy,
    compose_connector_write,
    compose_tools_write,
)
from tools.connectors.portal.wire import ConnectorCatalogResponse, ConnectorPolicyResponse


@pytest.mark.parametrize(
    ("body", "compose", "change", "expected"),
    [
        (None, compose_connector_write, SimpleNamespace(connector="linear", enabled=False),
         {"scope": "member", "disabledConnectors": ["linear"]}),
        (None, compose_tools_write, SimpleNamespace(connector="linear", disabled_tools=["create"]),
         {"scope": "member", "disabledConnectors": [], "tools": {"linear": ["create"]}}),
        (SimpleNamespace(mode="deny-all"), compose_connector_write,
         SimpleNamespace(connector="linear", enabled=True),
         {"scope": "member", "connectors": ["linear"]}),
        (SimpleNamespace(mode="deny-all"), compose_tools_write,
         SimpleNamespace(connector="linear", disabled_tools=["create"]),
         InvalidMemberPolicy),
        (SimpleNamespace(mode="allow", connectors=["linear"], tools={}), compose_connector_write,
         SimpleNamespace(connector="linear", enabled=False),
         {"scope": "member", "connectors": []}),
        (SimpleNamespace(mode="allow", connectors=["linear"], tools={}), compose_tools_write,
         SimpleNamespace(connector="linear", disabled_tools=["create"]),
         {"scope": "member", "connectors": ["linear"], "tools": {"linear": ["create"]}}),
        (SimpleNamespace(mode="deny", disabled_connectors=["linear"], tools={}), compose_connector_write,
         SimpleNamespace(connector="linear", enabled=True),
         {"scope": "member", "disabledConnectors": []}),
        (SimpleNamespace(mode="deny", disabled_connectors=["linear"], tools={}), compose_tools_write,
         SimpleNamespace(connector="linear", disabled_tools=["create"]),
         InvalidMemberPolicy),
        (SimpleNamespace(mode="deny", disabled_connectors=[], tools={}), compose_tools_write,
         SimpleNamespace(connector="linear", disabled_tools=["create"]),
         {"scope": "member", "disabledConnectors": [], "tools": {"linear": ["create"]}}),
    ],
)
def test_member_policy_writes_handle_every_member_mode_and_change(body, compose, change, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            compose(body, change)
        return
    assert compose(body, change) == expected


def test_catalog_and_policy_wires_tolerate_added_fields_and_keep_open_categories():
    catalog = ConnectorCatalogResponse.model_validate({
        "connectors": [{
            "slug": "linear", "name": "Linear", "description": "Issues", "category": "Future category",
            "logoUrl": "https://example.test/logo", "added": "later",
        }],
        "added": True,
    })
    policy = ConnectorPolicyResponse.model_validate({
        "layers": [
            {"kind": "org", "id": None, "revision": "1", "body": {"mode": "unrestricted", "added": True}},
            {"kind": "role", "id": "role", "revision": "2", "body": {"mode": "deny-all", "added": True}},
            {"kind": "member", "id": "member", "revision": "3", "body": {
                "mode": "allow", "connectors": ["linear"], "tools": {"linear": {"disable": ["create"], "added": True}},
                "tags": {"enable": ["readOnlyHint"], "added": True}, "added": True,
            }},
            {"kind": "member", "id": "member", "revision": "4", "body": {
                "mode": "deny", "disabledConnectors": ["linear"], "tools": {"linear": {"disable": ["create"]}},
            }},
        ],
        "effective": {"ignored": True},
    })

    assert catalog.connectors[0].category == "Future category"
    assert [layer.body.mode for layer in policy.layers] == ["unrestricted", "deny-all", "allow", "deny"]
    assert policy.layers[2].body.tools["linear"].disable == ["create"]
