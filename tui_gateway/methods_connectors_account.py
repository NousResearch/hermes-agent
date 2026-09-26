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
            except GatewayAuthError as exc:
                return _connector_auth_error(rid, exc)
            except Exception as exc:
                if getattr(exc, "code", None) == "org_required":
                    return _connector_rpc_error(
                        rid, 4090, ConnectorErrorReason.org_required, "Select an organization to manage connector rules."
                    )
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
    from tools.connectors.portal.errors import InvalidConnectorSlug
    from tools.connectors.portal.tools_cache import read_tools
    from tui_gateway.contracts.connectors import ConnectorErrorReason, ConnectorToolsResult

    client = PortalConnectorClient()
    client.require_authentication()
    try:
        listing = read_tools(request.slug, client=client, refresh=request.refresh)
    except (InvalidConnectorSlug, GatewayUnavailable):
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
    from tools.connectors.portal.client import PortalConnectorClient
    from tui_gateway.contracts.connectors import ConnectorAccountRow, ConnectorAccountsResult

    accounts = PortalConnectorClient().list_accounts()
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
    from tools.connectors.gateway.errors import GatewayAuthError, GatewayUnavailable, ToolGatewayError
    from tools.connectors.portal.client import PortalConnectorClient
    from tui_gateway.contracts.connectors import ConnectorAccountsRemoveResult, ConnectorErrorReason

    try:
        removed = PortalConnectorClient().delete_account(request.connection_id)
    except GatewayUnavailable as exc:
        if exc.code == "connection_not_found":
            return _connector_rpc_error(rid, 4041, ConnectorErrorReason.connection_not_found, "Connector account not found.")
        return _connector_rpc_error(rid, 5034, ConnectorErrorReason.accounts_unavailable, "Connector accounts are unavailable.")
    except GatewayAuthError:
        raise
    except ToolGatewayError as exc:
        if exc.code == "org_required":
            raise
        if exc.code == "invalid_connection_id":
            return _connector_rpc_error(rid, 4000, ConnectorErrorReason.invalid_params, "Connection id is invalid.")
        return _connector_rpc_error(rid, 5034, ConnectorErrorReason.accounts_unavailable, "Connector accounts are unavailable.")
    result = ConnectorAccountsRemoveResult(
        connection_id=removed["connectionId"], connector=removed["connector"], status=removed["status"]
    )
    return _ok(rid, result.model_dump(mode="json"))


@method("connectors.policy.get")
@_profile_scoped
@_account_method(
    unavailable=ConnectorErrorReason.policy_unavailable,
    unavailable_message="Connector policy is unavailable.",
)
def _(rid, _params):
    from tools.connectors.gateway.errors import GatewayAuthError, ToolGatewayError
    from tools.connectors.portal.client import PortalConnectorClient
    from tui_gateway.contracts.connectors import ConnectorErrorReason, ConnectorPolicyGetResult, ConnectorPolicyLayer

    try:
        client = PortalConnectorClient()
        client.require_authentication()
        policy = client.policy()
    except GatewayAuthError as exc:
        return _connector_rpc_error(rid, *_policy_auth_error(exc, ConnectorErrorReason))
    except ToolGatewayError as exc:
        return _connector_rpc_error(rid, *_policy_error(exc, ConnectorErrorReason))
    result = ConnectorPolicyGetResult(
        layers=[
            ConnectorPolicyLayer(kind=layer.kind, revision=layer.revision, body=_contract_policy_body(layer.body))
            for layer in policy.layers
        ],
        effective=_contract_policy_effective(policy.effective),
    )
    return _ok(rid, result.model_dump(mode="json"))


@method("connectors.policy.set")
@_profile_scoped
@_account_method(
    ConnectorPolicySetParams,
    invalid="Connector parameters are invalid.",
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
        body["expectedRevision"] = request.expected_revision
        result = client.set_policy(body)
    except GatewayAuthError as exc:
        return _connector_rpc_error(rid, *_policy_auth_error(exc, ConnectorErrorReason))
    except InvalidMemberPolicy:
        return _connector_rpc_error(rid, 4000, ConnectorErrorReason.invalid_policy, "Connector policy change is invalid.")
    except ToolGatewayError as exc:
        return _connector_rpc_error(rid, *_policy_error(exc, ConnectorErrorReason))
    return _ok(rid, ConnectorPolicySetResult(
        revision=result.revision, effective=_contract_policy_effective(result.effective)
    ).model_dump(mode="json"))


def _policy_rule_fields(body):
    from tui_gateway.contracts.connectors import ConnectorPolicyTags

    if body.mode in ("unrestricted", "deny-all"):
        return {}
    tags = None if body.tags is None else ConnectorPolicyTags(enable=body.tags.enable, disable=body.tags.disable)
    rules = {"tools": {connector: rule.disable for connector, rule in body.tools.items()}, "tags": tags}
    if body.mode == "allow":
        return {"connectors": body.connectors, **rules}
    return {"disabled_connectors": body.disabled_connectors, **rules}


def _contract_policy_body(body):
    from tui_gateway.contracts import connectors as contract

    models = {
        "unrestricted": contract.ConnectorPolicyUnrestrictedBody,
        "deny-all": contract.ConnectorPolicyDenyAllBody,
        "allow": contract.ConnectorPolicyAllowBody,
        "deny": contract.ConnectorPolicyDenyBody,
    }
    return models[body.mode](mode=body.mode, **_policy_rule_fields(body))


def _contract_policy_effective(effective):
    from tui_gateway.contracts import connectors as contract

    models = {
        "unrestricted": contract.ConnectorPolicyEffectiveUnrestricted,
        "deny-all": contract.ConnectorPolicyEffectiveDenyAll,
        "allow": contract.ConnectorPolicyEffectiveAllow,
        "deny": contract.ConnectorPolicyEffectiveDeny,
    }
    return models[effective.mode](
        mode=effective.mode, version=effective.version, revision=effective.revision,
        issued_at_ms=effective.issued_at_ms, **_policy_rule_fields(effective),
    )


def _policy_auth_error(exc, reasons):
    if exc.code in {"no_access", "ORG_ACCESS_DENIED"}:
        return 4030, reasons.org_access_denied, "This account cannot manage connectors for this organization."
    if exc.code == "forbidden":
        return 4030, reasons.forbidden_scope, "Connector policy cannot be changed for this member."
    if exc.status == 401 or exc.code in {"invalid_token", "INVALID_TOKEN", "NO_TOKEN"}:
        return 4032, reasons.needs_nous_auth, "Sign in to use connectors."
    return 5034, reasons.policy_unavailable, "Connector policy is unavailable."


def _policy_error(exc, reasons):
    by_code = {
        "policy_changed": (4090, reasons.policy_conflict, "Connector policy changed. Refresh and try again."),
        "org_required": (4090, reasons.org_required, "Select an organization to manage connector rules."),
        "forbidden": (4030, reasons.forbidden_scope, "Connector policy cannot be changed for this member."),
        "no_access": (4030, reasons.org_access_denied, "This account cannot manage connectors for this organization."),
        "invalid_connector_policy": (4000, reasons.invalid_policy, "Connector policy change is invalid."),
    }
    if exc.code in by_code:
        return by_code[exc.code]
    if exc.code == f"HTTP_{exc.status}" and exc.status == 400:
        return 4000, reasons.invalid_policy, "Connector policy change is invalid."
    return 5034, reasons.policy_unavailable, "Connector policy is unavailable."


def register(server):
    bind_module(globals(), server, skip=("_",))
    server._LONG_HANDLERS = server._LONG_HANDLERS | _registry.names()
