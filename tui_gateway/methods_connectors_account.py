"""Profile-scoped connector RPCs that are not owned by a chat session."""

from tui_gateway.contracts.connectors import (
    ConnectorAccountsParams,
    ConnectorAccountsRemoveParams,
    ConnectorErrorReason,
    ConnectorPolicySetParams,
    ConnectorToolsParams,
)

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped


def _account_method(params_model=None, *, invalid="", invalid_reason=ConnectorErrorReason.invalid_params,
                    unavailable, unavailable_message):
    """Gate, validate, then answer: a shut gate is 4031, bad params 4000, an auth failure 4032, anything else 5034."""
    def decorate(fn):
        def handler(rid, params):
            from pydantic import ValidationError
            from tools.connectors import connectors_available
            from tools.connectors.gateway.errors import GatewayAuthError
            from tui_gateway.contracts.connectors import ConnectorErrorReason

            if not connectors_available():
                return _connector_rpc_error(
                    rid, 4031, ConnectorErrorReason.connectors_unavailable, "Connectors are not available."
                )
            try:
                request = params if params_model is None else params_model.model_validate(params)
            except ValidationError:
                return _connector_rpc_error(rid, 4000, invalid_reason, invalid)
            try:
                return fn(rid, request)
            except GatewayAuthError:
                return _connector_rpc_error(rid, 4032, ConnectorErrorReason.needs_nous_auth, "Sign in to use connectors.")
            except Exception:
                return _connector_rpc_error(rid, 5034, unavailable, unavailable_message)

        return handler

    return decorate


@method("connectors.tools")
@_profile_scoped
@_account_method(
    ConnectorToolsParams,
    invalid="slug and refresh are required parameters",
    unavailable=ConnectorErrorReason.tools_unavailable,
    unavailable_message="Connector tools are unavailable.",
)
def _(rid, request):
    from tools.connectors.gateway.errors import GatewayUnavailable
    from tools.connectors.portal.client import PortalConnectorClient
    from tools.connectors.portal.tools_cache import read_tools
    from tui_gateway.contracts.connectors import ConnectorErrorReason, ConnectorToolsResult

    client = PortalConnectorClient()
    client.require_authentication()
    try:
        listing = read_tools(request.slug, client=client, refresh=request.refresh)
    except GatewayUnavailable:
        return _connector_rpc_error(rid, 4041, ConnectorErrorReason.connector_not_found, "Connector not found.")
    return _ok(rid, ConnectorToolsResult.model_validate(listing, from_attributes=True).model_dump(mode="json"))


@method("connectors.catalog")
@_profile_scoped
@_account_method(
    unavailable=ConnectorErrorReason.catalog_unavailable,
    unavailable_message="Connector catalog is unavailable.",
)
def _(rid, _params):
    from tools.connectors.portal.client import PortalConnectorClient
    from tui_gateway.contracts.connectors import ConnectorsCatalogResult

    client = PortalConnectorClient()
    client.require_authentication()
    catalog = client.catalog()
    return _ok(rid, ConnectorsCatalogResult.model_validate(catalog, from_attributes=True).model_dump(mode="json"))


@method("connectors.accounts")
@_profile_scoped
@_account_method(
    ConnectorAccountsParams,
    invalid="connector must be a slug.",
    unavailable=ConnectorErrorReason.accounts_unavailable,
    unavailable_message="Connector accounts are unavailable.",
)
def _(rid, request):
    from tools.connectors.gateway.client import ConnectorClient
    from tui_gateway.contracts.connectors import ConnectorAccountRow, ConnectorAccountsResult

    accounts = ConnectorClient().list_accounts()
    rows = [account for account in accounts if request.connector is None or account["connector"] == request.connector]
    result = ConnectorAccountsResult(accounts=[ConnectorAccountRow(
        connection_id=account["connectionId"],
        connector=account["connector"],
        status=account["status"],
        status_reason=account.get("statusReason"),
        label=account["label"],
        alias=account.get("alias"),
        active=account["active"],
        created_at=account["createdAt"],
        updated_at=account["updatedAt"],
    ) for account in rows])
    return _ok(rid, result.model_dump(mode="json"))


@method("connectors.accounts.remove")
@_profile_scoped
@_account_method(
    ConnectorAccountsRemoveParams,
    invalid="connection_id is required.",
    unavailable=ConnectorErrorReason.accounts_unavailable,
    unavailable_message="Connector accounts are unavailable.",
)
def _(rid, request):
    from tools.connectors.gateway.client import ConnectorClient
    from tools.connectors.gateway.errors import GatewayUnavailable
    from tui_gateway.contracts.connectors import ConnectorAccountsRemoveResult, ConnectorErrorReason

    try:
        removed = ConnectorClient().delete_account(request.connection_id)
    except GatewayUnavailable as exc:
        # Only the gateway's own "no such account" removes the card; any other failure is an outage.
        if exc.code == "connection_not_found":
            return _connector_rpc_error(rid, 4041, ConnectorErrorReason.connection_not_found, "Connector account not found.")
        return _connector_rpc_error(rid, 5034, ConnectorErrorReason.accounts_unavailable, "Connector accounts are unavailable.")
    result = ConnectorAccountsRemoveResult(connection_id=removed["connectionId"], status=removed["status"])
    return _ok(rid, result.model_dump(mode="json"))


@method("connectors.policy.get")
@_profile_scoped
@_account_method(
    unavailable=ConnectorErrorReason.policy_unavailable,
    unavailable_message="Connector policy is unavailable.",
)
def _(rid, _params):
    from tools.connectors.portal.client import PortalConnectorClient
    from tui_gateway.contracts.connectors import (
        ConnectorPolicyAllowBody,
        ConnectorPolicyDenyAllBody,
        ConnectorPolicyDenyBody,
        ConnectorPolicyGetResult,
        ConnectorPolicyLayer,
        ConnectorPolicyTags,
        ConnectorPolicyUnrestrictedBody,
    )

    client = PortalConnectorClient()
    client.require_authentication()
    policy = client.policy()
    result = ConnectorPolicyGetResult(layers=[
        ConnectorPolicyLayer(kind=layer.kind, revision=layer.revision, body=_contract_policy_body(
            layer.body,
            ConnectorPolicyAllowBody,
            ConnectorPolicyDenyAllBody,
            ConnectorPolicyDenyBody,
            ConnectorPolicyTags,
            ConnectorPolicyUnrestrictedBody,
        ))
        for layer in policy.layers
    ])
    return _ok(rid, result.model_dump(mode="json"))


@method("connectors.policy.set")
@_profile_scoped
@_account_method(
    ConnectorPolicySetParams,
    invalid="Connector policy change is invalid.",
    invalid_reason=ConnectorErrorReason.invalid_policy,
    unavailable=ConnectorErrorReason.policy_unavailable,
    unavailable_message="Connector policy is unavailable.",
)
def _(rid, request):
    from tools.connectors.gateway.errors import GatewayAuthError, ToolGatewayError
    from tools.connectors.portal.client import PortalConnectorClient
    from tools.connectors.portal.policy import InvalidMemberPolicy, compose_connector_write, compose_tools_write
    from tui_gateway.contracts.connectors import ConnectorErrorReason, ConnectorPolicySetResult, ToolsChange

    try:
        client = PortalConnectorClient()
        client.require_authentication()
        policy = client.policy()
        member = next((layer.body for layer in policy.layers if layer.kind == "member"), None)
        compose = compose_tools_write if isinstance(request.change, ToolsChange) else compose_connector_write
        body = compose(member, request.change)
        if request.expected_revision is not None:
            body["expectedRevision"] = request.expected_revision
        result = client.set_policy(body)
    except GatewayAuthError as exc:
        if exc.status == 403:
            return _connector_rpc_error(rid, 4030, ConnectorErrorReason.forbidden_scope, "Connector policy cannot be changed for this member.")
        return _connector_rpc_error(rid, 4032, ConnectorErrorReason.needs_nous_auth, "Sign in to use connectors.")
    except InvalidMemberPolicy:
        return _connector_rpc_error(rid, 4000, ConnectorErrorReason.invalid_policy, "Connector policy change is invalid.")
    except ToolGatewayError as exc:
        error = _policy_error(exc, ConnectorErrorReason)
        if error is not None:
            return _connector_rpc_error(rid, *error)
        return _connector_rpc_error(rid, 5034, ConnectorErrorReason.policy_unavailable, "Connector policy is unavailable.")
    return _ok(rid, ConnectorPolicySetResult(revision=result.revision).model_dump(mode="json"))


def _contract_policy_body(body, allow, deny_all, deny, tags, unrestricted):
    if body.mode == "unrestricted":
        return unrestricted(mode=body.mode)
    if body.mode == "deny-all":
        return deny_all(mode=body.mode)
    rendered_tags = None if body.tags is None else tags(enable=body.tags.enable, disable=body.tags.disable)
    tools = {connector: rule.disable for connector, rule in body.tools.items()}
    if body.mode == "allow":
        return allow(mode=body.mode, connectors=body.connectors, tools=tools, tags=rendered_tags)
    return deny(mode=body.mode, disabled_connectors=body.disabled_connectors, tools=tools, tags=rendered_tags)


def _policy_error(exc, reasons):
    table = {
        400: (4000, reasons.invalid_policy, "Connector policy change is invalid."),
        403: (4030, reasons.forbidden_scope, "Connector policy cannot be changed for this member."),
        409: (4090, reasons.policy_conflict, "Connector policy changed. Refresh and try again."),
    }
    return table.get(exc.status)


def register(server):
    bind_module(globals(), server, skip=("_",))
    # Portal and tool gateway reads and writes can block, so these RPCs must run off the server loop.
    server._LONG_HANDLERS = server._LONG_HANDLERS | _registry.names()
