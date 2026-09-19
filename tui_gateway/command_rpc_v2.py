"""Versioned command RPC handlers and legacy compatibility shims.

The owning gateway module is intentionally not imported here.  Callers provide
its ``_ok``/``_err`` envelope functions and install the returned bounded
handlers into the method registry.  This keeps the v2 ABI testable without
adding another command table or growing an existing godfile.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping, MutableMapping
from types import MappingProxyType
from typing import Any

from hermes_cli.command_dispatcher import (
    CommandDispatcher,
    CommandInvocation,
    catalog_revision,
    catalog_to_dict,
)

CATALOG_V2_METHOD = "commands.catalog.v2"
INVOKE_METHOD = "commands.invoke"
LEGACY_CATALOG_METHOD = "commands.catalog"
LEGACY_DISPATCH_METHOD = "command.dispatch"
LEGACY_SLASH_EXEC_METHOD = "slash.exec"


def _row_value(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def _resolve_catalog_token(snapshot: Any, token: str) -> Any | None:
    normalized = str(token or "").strip().lstrip("/").casefold()
    if not normalized:
        return None

    resolver = getattr(snapshot, "resolve_command", None)
    if callable(resolver):
        return resolver(normalized)
    resolver = getattr(snapshot, "resolve", None)
    if callable(resolver):
        return resolver(normalized)

    commands = getattr(snapshot, "commands", ())
    rows = commands.values() if isinstance(commands, Mapping) else commands
    for row in rows or ():
        name = str(_row_value(row, "name", "") or "").casefold()
        aliases = tuple(_row_value(row, "aliases", ()) or ())
        if name == normalized or any(str(alias).casefold() == normalized for alias in aliases):
            return row
    return None


def _command_id(row: Any) -> str:
    return str(_row_value(row, "command_id", "") or "").strip()


def _legacy_invocation(
    *,
    snapshot: Any,
    token: str,
    raw_arguments: str,
    raw_input: str,
    params: Mapping[str, Any],
    source_method: str,
) -> CommandInvocation:
    row = _resolve_catalog_token(snapshot, token)
    command_id = _command_id(row) if row is not None else f"unknown.{token.lstrip('/')}"
    return CommandInvocation(
        request_id=str(params.get("request_id") or f"{source_method}:{uuid.uuid4()}"),
        catalog_revision=catalog_revision(snapshot),
        command_id=command_id,
        entered_name=str(token or "").lstrip("/"),
        raw_input=raw_input,
        raw_arguments=raw_arguments,
        parsed_arguments={"legacy_raw": raw_arguments},
        surface=str(params.get("surface") or "legacy-rpc"),
        platform=str(params.get("platform") or "tui"),
        actor=params.get("actor") or {},
        profile_home=str(params.get("profile_home") or ""),
        cwd=str(params.get("cwd") or ""),
        session_id=str(params.get("session_id") or ""),
        chat_id=str(params.get("chat_id") or ""),
        channel_id=str(params.get("channel_id") or ""),
        thread_id=str(params.get("thread_id") or ""),
        source_id=str(params.get("source_id") or ""),
        locale=str(params.get("locale") or ""),
        attachments=tuple(params.get("attachments") or ()),
        capabilities=params.get("capabilities") or {},
        idempotency_key=str(params.get("idempotency_key") or ""),
        retry_of=str(params.get("retry_of") or ""),
    )


class CommandRPCV2:
    """JSON-RPC-compatible adapter around one canonical dispatcher."""

    def __init__(
        self,
        *,
        catalog_provider: Callable[[Mapping[str, Any]], Any],
        dispatcher: CommandDispatcher,
        ok: Callable[[Any, Mapping[str, Any]], dict[str, Any]],
        err: Callable[[Any, int, str], dict[str, Any]],
        legacy_catalog_projector: Callable[[Any], Mapping[str, Any]] | None = None,
        legacy_result_projector: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> None:
        self._catalog_provider = catalog_provider
        self._dispatcher = dispatcher
        self._ok = ok
        self._err = err
        self._legacy_catalog_projector = legacy_catalog_projector
        self._legacy_result_projector = legacy_result_projector

    def catalog_v2(self, rid: Any, params: Mapping[str, Any]) -> dict[str, Any]:
        try:
            snapshot = self._catalog_provider(params)
            return self._ok(rid, catalog_to_dict(snapshot))
        except (TypeError, ValueError) as exc:
            return self._err(rid, 4003, str(exc))
        except Exception as exc:
            return self._err(rid, 5020, str(exc))

    def invoke(self, rid: Any, params: Mapping[str, Any]) -> dict[str, Any]:
        try:
            raw = params.get("invocation", params)
            invocation = CommandInvocation.from_mapping(raw)
        except (TypeError, ValueError) as exc:
            return self._err(rid, 4003, str(exc))
        return self._ok(rid, self._dispatcher.dispatch(invocation).to_dict())

    def legacy_catalog(self, rid: Any, params: Mapping[str, Any]) -> dict[str, Any]:
        if self._legacy_catalog_projector is None:
            return self._err(rid, 5018, "legacy catalog projector is not installed")
        try:
            snapshot = self._catalog_provider(params)
            projected = self._legacy_catalog_projector(snapshot)
            return self._ok(rid, dict(projected))
        except Exception as exc:
            return self._err(rid, 5020, str(exc))

    def legacy_dispatch(self, rid: Any, params: Mapping[str, Any]) -> dict[str, Any]:
        token = str(params.get("name") or "").strip().lstrip("/")
        raw_arguments = str(params.get("arg") or "")
        if not token:
            return self._err(rid, 4004, "empty command")
        raw_input = f"/{token}" + (f" {raw_arguments}" if raw_arguments else "")
        return self._run_legacy(
            rid,
            params,
            token=token,
            raw_arguments=raw_arguments,
            raw_input=raw_input,
            source_method=LEGACY_DISPATCH_METHOD,
        )

    def legacy_slash_exec(self, rid: Any, params: Mapping[str, Any]) -> dict[str, Any]:
        raw_input = str(params.get("command") or "").strip()
        if not raw_input:
            return self._err(rid, 4004, "empty command")
        text = raw_input[1:] if raw_input.startswith("/") else raw_input
        token, _, raw_arguments = text.partition(" ")
        if not token or "/" in token:
            return self._err(rid, 4004, "invalid command")
        return self._run_legacy(
            rid,
            params,
            token=token,
            raw_arguments=raw_arguments,
            raw_input=raw_input,
            source_method=LEGACY_SLASH_EXEC_METHOD,
        )

    def _run_legacy(
        self,
        rid: Any,
        params: Mapping[str, Any],
        *,
        token: str,
        raw_arguments: str,
        raw_input: str,
        source_method: str,
    ) -> dict[str, Any]:
        if self._legacy_result_projector is None:
            return self._err(rid, 5018, "legacy result projector is not installed")
        try:
            snapshot = self._catalog_provider(params)
            invocation = _legacy_invocation(
                snapshot=snapshot,
                token=token,
                raw_arguments=raw_arguments,
                raw_input=raw_input,
                params=params,
                source_method=source_method,
            )
            result = self._dispatcher.dispatch(invocation).to_dict()
            return self._ok(rid, dict(self._legacy_result_projector(result)))
        except (TypeError, ValueError) as exc:
            return self._err(rid, 4003, str(exc))
        except Exception as exc:
            return self._err(rid, 5020, str(exc))

    def handlers(self, *, include_legacy: bool = False) -> Mapping[str, Callable]:
        handlers: dict[str, Callable] = {
            CATALOG_V2_METHOD: self.catalog_v2,
            INVOKE_METHOD: self.invoke,
        }
        if include_legacy:
            handlers.update(
                {
                    LEGACY_CATALOG_METHOD: self.legacy_catalog,
                    LEGACY_DISPATCH_METHOD: self.legacy_dispatch,
                    LEGACY_SLASH_EXEC_METHOD: self.legacy_slash_exec,
                }
            )
        return MappingProxyType(handlers)

    def install(
        self,
        methods: MutableMapping[str, Callable],
        *,
        include_legacy: bool = False,
        replace_existing: bool = False,
    ) -> None:
        handlers = self.handlers(include_legacy=include_legacy)
        collisions = sorted(name for name in handlers if name in methods)
        if collisions and not replace_existing:
            joined = ", ".join(collisions)
            raise ValueError(f"command RPC method collision: {joined}")
        methods.update(handlers)
