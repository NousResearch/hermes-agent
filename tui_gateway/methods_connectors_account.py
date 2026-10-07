"""Account-scoped connector RPCs.

Every method in this module talked to the hosted connector portal, which is not
part of this build. The methods stay registered so older frontends get a clean
"unavailable" answer instead of a transport error.
"""

from tui_gateway.contracts.connectors import ConnectorErrorReason

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped


def _unavailable(rid, reason, message):
    return _connector_rpc_error(rid, 4031, reason, message)


@method("connectors.tools")
@_profile_scoped
def _(rid, _params):
    return _unavailable(rid, ConnectorErrorReason.tools_unavailable, "Connector tools are unavailable in this build.")


@method("connectors.catalog")
@_profile_scoped
def _(rid, _params):
    return _unavailable(rid, ConnectorErrorReason.catalog_unavailable, "The hosted connector catalog is unavailable in this build.")


@method("connectors.accounts")
@_profile_scoped
def _(rid, _params):
    return _unavailable(rid, ConnectorErrorReason.accounts_unavailable, "Connector accounts are unavailable in this build.")


@method("connectors.accounts.remove")
@_profile_scoped
def _(rid, _params):
    return _unavailable(rid, ConnectorErrorReason.accounts_unavailable, "Connector accounts are unavailable in this build.")


@method("connectors.policy.get")
@_profile_scoped
def _(rid, _params):
    return _unavailable(rid, ConnectorErrorReason.policy_unavailable, "Connector policy is unavailable in this build.")


@method("connectors.policy.set")
@_profile_scoped
def _(rid, _params):
    return _unavailable(rid, ConnectorErrorReason.policy_unavailable, "Connector policy is unavailable in this build.")


def register(server):
    bind_module(globals(), server, skip=("_",))
    server._LONG_HANDLERS = server._LONG_HANDLERS | _registry.names()
