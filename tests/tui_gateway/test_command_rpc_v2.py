from __future__ import annotations

from dataclasses import dataclass

import pytest

from hermes_cli.command_dispatcher import (
    CommandBinding,
    CommandDispatcher,
    CommandPolicyState,
    CommandResult,
)
from tui_gateway.command_rpc_v2 import (
    CATALOG_V2_METHOD,
    INVOKE_METHOD,
    LEGACY_CATALOG_METHOD,
    LEGACY_DISPATCH_METHOD,
    LEGACY_SLASH_EXEC_METHOD,
    CommandRPCV2,
)


@dataclass(frozen=True)
class Command:
    command_id: str = "session.new"
    name: str = "new"
    aliases: tuple[str, ...] = ("reset",)
    execution_owner: str = "server"
    handler_id: str = "session.new"


@dataclass(frozen=True)
class Catalog:
    revision: str = "rev-1"
    commands: tuple[Command, ...] = (Command(),)

    def resolve_command_id(self, command_id: str):
        return next(
            (row for row in self.commands if row.command_id == command_id),
            None,
        )

    def resolve(self, token: str):
        normalized = token.casefold()
        return next(
            (
                row
                for row in self.commands
                if row.name == normalized or normalized in row.aliases
            ),
            None,
        )

    def to_dict(self):
        return {
            "schema_version": 2,
            "revision": self.revision,
            "commands": [
                {
                    "command_id": row.command_id,
                    "name": row.name,
                    "aliases": list(row.aliases),
                }
                for row in self.commands
            ],
        }


def ok(rid, payload):
    return {"id": rid, "result": dict(payload)}


def err(rid, code, message):
    return {"id": rid, "error": {"code": code, "message": message}}


def build_rpc(calls=None, *, legacy=True):
    calls = calls if calls is not None else []
    catalog = Catalog()

    def handler(invocation, _command):
        calls.append(invocation)
        return CommandResult(
            status="ok",
            command_id=invocation.command_id,
            content=({"type": "text", "text": "created"},),
        )

    runtime = CommandDispatcher(
        catalog_provider=lambda _inv: catalog,
        policy_resolver=lambda *_args: CommandPolicyState(),
        bindings=(
            CommandBinding(
                command_id="session.new",
                execution_owner="server",
                handler_id="session.new",
                handler=handler,
            ),
        ),
    )
    return CommandRPCV2(
        catalog_provider=lambda _params: catalog,
        dispatcher=runtime,
        ok=ok,
        err=err,
        legacy_catalog_projector=(
            (lambda snapshot: {"revision": snapshot.revision, "pairs": [["/new", "New"]]})
            if legacy
            else None
        ),
        legacy_result_projector=(
            (lambda result: {"type": "exec", "output": result["content"][0]["text"]})
            if legacy
            else None
        ),
    )


def v2_params():
    return {
        "request_id": "request-1",
        "catalog_revision": "rev-1",
        "command_id": "session.new",
        "entered_name": "reset",
        "raw_input": "/reset now",
        "raw_arguments": "now",
        "parsed_arguments": {"name": "now"},
        "surface": "desktop",
    }


def test_v2_handlers_use_exact_public_method_names():
    handlers = build_rpc().handlers()

    assert tuple(handlers) == (CATALOG_V2_METHOD, INVOKE_METHOD)
    with pytest.raises(TypeError):
        handlers["other"] = lambda: None


def test_catalog_v2_returns_client_safe_snapshot():
    response = build_rpc().catalog_v2("rpc-1", {"surface": "desktop"})

    assert response["result"]["schema_version"] == 2
    assert response["result"]["revision"] == "rev-1"
    assert response["result"]["commands"][0]["command_id"] == "session.new"


def test_commands_invoke_parses_and_dispatches_typed_invocation():
    calls = []
    response = build_rpc(calls).invoke("rpc-1", {"invocation": v2_params()})

    assert response["result"]["status"] == "ok"
    assert response["result"]["catalog_revision"] == "rev-1"
    assert calls[0].entered_name == "reset"
    assert calls[0].raw_arguments == "now"


def test_commands_invoke_rejects_malformed_input_as_rpc_error():
    response = build_rpc().invoke("rpc-1", {"invocation": {"command_id": "x"}})

    assert response["error"]["code"] == 4003
    assert "request_id" in response["error"]["message"]


def test_legacy_command_dispatch_is_a_v2_invocation_shim():
    calls = []
    response = build_rpc(calls).legacy_dispatch(
        "rpc-1",
        {
            "name": "/reset",
            "arg": "now",
            "session_id": "session-1",
            "platform": "tui",
        },
    )

    invocation = calls[0]
    assert invocation.command_id == "session.new"
    assert invocation.entered_name == "reset"
    assert invocation.raw_input == "/reset now"
    assert invocation.catalog_revision == "rev-1"
    assert response["result"] == {"type": "exec", "output": "created"}


def test_legacy_slash_exec_is_the_same_canonical_shim():
    calls = []
    response = build_rpc(calls).legacy_slash_exec(
        "rpc-1",
        {"command": "/reset now", "session_id": "session-1"},
    )

    assert calls[0].command_id == "session.new"
    assert calls[0].raw_arguments == "now"
    assert response["result"]["output"] == "created"


def test_legacy_shims_fail_closed_without_explicit_projectors():
    rpc = build_rpc(legacy=False)

    assert rpc.legacy_catalog("rpc-1", {})["error"]["code"] == 5018
    assert (
        rpc.legacy_dispatch("rpc-1", {"name": "new"})["error"]["code"]
        == 5018
    )


def test_include_legacy_exports_all_three_compatibility_methods():
    handlers = build_rpc().handlers(include_legacy=True)

    assert set(handlers) == {
        CATALOG_V2_METHOD,
        INVOKE_METHOD,
        LEGACY_CATALOG_METHOD,
        LEGACY_DISPATCH_METHOD,
        LEGACY_SLASH_EXEC_METHOD,
    }


def test_install_refuses_method_collisions_unless_explicitly_replacing():
    rpc = build_rpc()
    methods = {CATALOG_V2_METHOD: object()}

    with pytest.raises(ValueError, match="command RPC method collision"):
        rpc.install(methods)

    rpc.install(methods, replace_existing=True)
    assert methods[CATALOG_V2_METHOD] == rpc.catalog_v2
    assert methods[INVOKE_METHOD] == rpc.invoke
