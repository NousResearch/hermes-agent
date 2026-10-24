"""Automatic admission of existing hosted read capabilities; never starts authorization.

Call in the owning profile/secret scope. No cache is an authority: discovery and every
execution reread policy, account inventory, facet metadata and argument schemas.
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import copy
import json
import time
from uuid import uuid4
from hashlib import sha256
from typing import Any, Callable

from hermes_constants import get_hermes_home
from tools.connectors.gateway.names import format_connector_name


@dataclass(frozen=True)
class ReadCapability:
    capability_id: str
    source_id: str
    name: str
    description: str
    parameters: dict[str, Any]
    connector: str
    connection_id: str
    alias: str | None


class ConnectedSources:
    """Profile-bound discovery, with injectable metadata transports for offline tests."""

    def __init__(self, *, client_factory: Callable | None = None,
                 portal_factory: Callable | None = None, availability: Callable | None = None):
        self._client_factory = client_factory
        self._portal_factory = portal_factory
        self._availability = availability
        self._home = str(get_hermes_home().resolve())
        self._profile_key = sha256(self._home.encode()).hexdigest()[:16]

    def discover(self, *, deadline: float | None = None) -> list[ReadCapability]:
        from tools.connectors.gateway.bridge import _default_client_factory
        from tools.connectors.gateway.config import connectors_available
        from tools.connectors.portal.client import PortalConnectorClient

        check_deadline(deadline)
        if str(get_hermes_home().resolve()) != self._home:
            return []
        if not (self._availability or connectors_available)():
            return []
        client = _deadline_client((self._client_factory or _default_client_factory)(), deadline)
        portal = _deadline_client((self._portal_factory or PortalConnectorClient)(), deadline)
        rows = client.list_connectors()
        check_deadline(deadline)
        accounts = portal.list_accounts()
        check_deadline(deadline)
        admitted = []
        for row in rows:
            if row.get("enabled") is not True or row.get("connected") is not True:
                continue
            active = _selectable_accounts(accounts, row["connector"])
            if active:
                admitted.extend(self._connector_capabilities(row, active, client, portal, deadline))
        return admitted

    def _connector_capabilities(self, row, accounts, client, portal, deadline):
        slug = row["connector"]
        check_deadline(deadline)
        listing = portal.tools(slug)
        check_deadline(deadline)
        if getattr(listing, "connector", None) != slug:
            return []
        tools = [t for t in listing.tools if t.facet == "read" and not t.deprecated
                 and t.slug not in row.get("disabledTools", ())]
        if not tools:
            return []
        schemas = client.schemas([t.slug for t in tools])["schemas"]
        check_deadline(deadline)
        admitted = []
        for tool in tools:
            schema = schemas.get(tool.slug, {})
            if schema.get("connector") != slug or schema.get("tool") != tool.slug:
                continue
            parameters = schema.get("input_schema")
            if not _supported_schema(parameters):
                continue
            for account in accounts:
                source_id = f"connector:{self._profile_key}:{slug}:{account['connectionId']}"
                admitted.append(ReadCapability(
                    capability_id=f"{source_id}:{tool.slug}", source_id=source_id,
                    name=format_connector_name(slug, tool.slug), description=tool.description,
                    parameters=parameters, connector=slug, connection_id=account["connectionId"],
                    alias=account.get("alias")))
        return admitted

    def read(self, capability_id: str, arguments: dict[str, Any], *,
             deadline: float | None = None, check_current: Callable | None = None) -> dict[str, Any]:
        """Validate the exact ID and schema, then run the normal policies before an exact read.

        The generic connector bridge retries a different literal tool slug on a miss and
        stages file inputs. Neither is permissible for unattended read-only assistance.
        Reuse its dispatch policies, but send only the classified slug without file I/O.
        """
        from tools.connectors.data_use import collect_data_use, record_data_use
        from tools.connectors.gateway.bridge import _default_client_factory
        from tools.connectors.gateway.merge import PlannedCall

        capability = next((c for c in self.discover(deadline=deadline) if c.capability_id == capability_id), None)
        if capability is None:
            raise ValueError("capability_unavailable")
        _validate_arguments(capability, arguments)
        pinned = dict(arguments)
        if capability.alias:
            pinned["connector_alias"] = capability.alias
        actual = []

        def execute(next_args):
            if actual:
                raise ValueError("read_already_executed")
            args = dict(next_args)
            if args.pop("connector_alias", None) != capability.alias:
                raise ValueError("account_changed")
            _validate_arguments(capability, args)
            # A policy hook may have revoked access or switched the owning profile.
            if capability not in self.discover(deadline=deadline):
                raise ValueError("capability_unavailable")
            client = _deadline_client((self._client_factory or _default_client_factory)(), deadline,
                                      check_current=check_current)
            plan = PlannedCall(0, capability.name, capability.connector,
                               capability.capability_id.rsplit(":", 1)[1], dict(next_args))
            check_deadline(deadline)
            if check_current is not None:
                check_current()
            result = client.execute([plan])[0]
            check_deadline(deadline)
            if result.get("error") or result.get("account") != capability.alias:
                raise ValueError("read_unavailable")
            record_data_use(result.get("data_use") or ())
            raw = json.dumps({"response": result["data"], "account": result.get("account")})
            actual.append(raw)
            return raw

        with collect_data_use() as requires:
            raw = _dispatch_read(capability.name, pinned, execute)
        if not actual or raw != actual[0]:
            raise ValueError("read_unavailable")
        return {"source_id": capability.source_id, "capability_id": capability.capability_id,
                "response": json.loads(raw)["response"], "data_use": sorted(requires)}

    def revalidate(self, evidence: list[dict], *, deadline: float | None = None) -> bool:
        """Current access, not freshness of record content. Recheck immediately before delivery."""
        try:
            current = {(c.source_id, c.capability_id) for c in self.discover(deadline=deadline)}
            return bool(evidence) and all(
                isinstance(e, dict) and (e.get("source_id"), e.get("capability_id")) in current
                for e in evidence)
        except Exception:  # health: allow BLE001 -- access revalidation fails closed without credential-bearing errors
            return False


def _selectable_accounts(accounts: list[dict], slug: str) -> list[dict]:
    active = [a for a in accounts if a["connector"] == slug and a["status"] == "active"
              and a.get("disabled") is False]
    # A named account is pinned by alias; an unnamed account must be the unique default.
    aliases = [a.get("alias") for a in active if a.get("alias")]
    if len(aliases) != len(set(aliases)):
        return []
    defaults = sum(a.get("active") is True for a in active)
    return [a for a in active if a.get("alias") or (a.get("active") is True and defaults == 1)]


def check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("deadline_exceeded")


class _DeadlineTransport:
    def __init__(self, transport, deadline: float | None, check_current: Callable | None = None):
        self.transport = transport
        self.deadline = deadline
        self.check_current = check_current

    def request(self, method, url, **kwargs):
        if self.check_current is not None:
            self.check_current()
        check_deadline(self.deadline)
        if self.deadline is not None:
            remaining = self.deadline - time.monotonic()
            kwargs["timeout"] = min(kwargs.get("timeout") or remaining, remaining)
        result = self.transport.request(method, url, **kwargs)
        check_deadline(self.deadline)
        return result


def _deadline_client(client, deadline: float | None, *, check_current: Callable | None = None):
    if deadline is None and check_current is None:
        return client
    # Client instances only: never change process-wide transports or other profiles.
    bounded = copy(client)
    bounded._transport = _DeadlineTransport(client._transport, deadline, check_current)
    if deadline is not None and hasattr(client, "_sleep"):
        def wait(seconds):
            check_deadline(deadline)
            if seconds >= deadline - time.monotonic():
                raise TimeoutError("deadline_exceeded")
            client._sleep(seconds)
            check_deadline(deadline)
        bounded._sleep = wait
    return bounded


def _safe_schema_nodes(node: Any) -> bool:
    if isinstance(node, list):
        return all(_safe_schema_nodes(child) for child in node)
    if not isinstance(node, dict):
        return True
    if node.get("file_uploadable") or "$dynamicRef" in node or "$recursiveRef" in node:
        return False
    if "$ref" in node and (not isinstance(node["$ref"], str) or not node["$ref"].startswith("#/")):
        return False
    return all(_safe_schema_nodes(child) for child in node.values())


def _supported_schema(schema: Any) -> bool:
    if not isinstance(schema, dict) or schema.get("type") != "object" or not _safe_schema_nodes(schema):
        return False
    try:
        from jsonschema.exceptions import SchemaError
        from jsonschema.validators import validator_for
    except ImportError:
        return _subset_schema(schema)
    try:
        if "$schema" in schema and validator_for(schema, default=None) is None:
            return False
        validator_for(schema).check_schema(schema)
        return True
    except (SchemaError, ValueError, TypeError, RecursionError):
        return False


def _validate_arguments(capability: ReadCapability, arguments: Any) -> None:
    if not isinstance(arguments, dict) or "connector_alias" in arguments:
        raise ValueError("invalid_arguments")
    json.dumps(arguments, allow_nan=False)
    try:
        from jsonschema.validators import validator_for
    except ImportError:
        if not _subset_schema(capability.parameters) or not _subset_valid(capability.parameters, arguments):
            raise ValueError("invalid_arguments")
        return
    validator = validator_for(capability.parameters)
    validator.check_schema(capability.parameters)
    if not validator(capability.parameters).is_valid(arguments):
        raise ValueError("invalid_arguments")


_SUBSET_KEYS = frozenset({"type", "properties", "required", "additionalProperties", "items", "enum",
                          "title", "description", "default"})
_JSON_TYPES = {
    "object": lambda v: isinstance(v, dict), "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str), "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: not isinstance(v, bool) and (isinstance(v, int) or isinstance(v, float) and v.is_integer()),
}


def _subset_schema(schema) -> bool:
    """Strict dependency-free subset: unsupported constraints are never ignored.

    Hosted connections do not require the optional MCP/jsonschema installation.
    Complex schemas remain unavailable there instead of skipping their validation.
    """
    if isinstance(schema, bool):
        return True
    if not isinstance(schema, dict) or set(schema) - _SUBSET_KEYS:
        return False
    if "type" in schema and schema["type"] not in _JSON_TYPES:
        return False
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    enum = schema.get("enum", [None])
    if not isinstance(properties, dict) or not isinstance(required, list):
        return False
    if not all(isinstance(k, str) for k in required) or not isinstance(enum, list) or not enum:
        return False
    children = [*properties.values(), schema.get("items", True), schema.get("additionalProperties", True)]
    return all(_subset_schema(child) for child in children)


def _subset_valid(schema, value) -> bool:
    if isinstance(schema, bool):
        return schema
    if "type" in schema and not _JSON_TYPES[schema["type"]](value):
        return False
    if "enum" in schema and not any(type(v) is type(value) and v == value for v in schema["enum"]):
        return False
    if isinstance(value, dict):
        if not set(schema.get("required", [])) <= value.keys():
            return False
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        return all(_subset_valid(properties.get(k, additional), v) for k, v in value.items())
    if isinstance(value, list):
        return all(_subset_valid(schema.get("items", True), v) for v in value)
    return True


def _dispatch_read(name: str, arguments: dict, execute: Callable) -> str:
    """Same hook/middleware chain as model_tools, with a constrained final executor."""
    import model_tools
    from hermes_cli.middleware import run_tool_execution_middleware

    ids = model_tools._CallIds(tool_call_id=f"assistance-{uuid4().hex}")
    start = time.monotonic()
    args, original, trace = model_tools._apply_request_middleware(name, arguments, ids, [])
    args, blocked = model_tools._pre_dispatch_guards(name, args, False, ids, trace)
    if blocked:
        result, error_type, error_message = blocked
        model_tools._emit_post_tool_call_hook(
            function_name=name, function_args=args, result=result, tool_call_id=ids.tool_call_id,
            status="blocked", error_type=error_type, error_message=error_message, middleware_trace=trace)
        return result
    with model_tools._approval_observability(ids):
        result = run_tool_execution_middleware(name, args, execute, original_args=original, **ids.hook_kwargs())
    duration = int((time.monotonic() - start) * 1000)
    model_tools._emit_post_tool_call_hook(
        function_name=name, function_args=args, result=result, tool_call_id=ids.tool_call_id,
        duration_ms=duration, middleware_trace=trace)
    return model_tools._apply_transform_tool_result_hook(name, args, result, duration, ids)
