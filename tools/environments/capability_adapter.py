"""Map verified image capabilities to Hermes model-tool names."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class CapabilityDefinition:
    """Tool names exposed only when all required capabilities are available."""

    name: str
    tools: frozenset[str]
    requires: frozenset[str]


CAPABILITIES: dict[str, CapabilityDefinition] = {
    "terminal": CapabilityDefinition("terminal", frozenset({"terminal"}), frozenset({"terminal"})),
    "files": CapabilityDefinition(
        "files",
        frozenset({"read_file", "write_file", "patch", "search_files"}),
        frozenset({"files"}),
    ),
    "computer_use": CapabilityDefinition(
        "computer_use", frozenset({"computer_use"}), frozenset({"computer_use"})
    ),
}


def resolve_tools(
    available_capabilities: Iterable[str],
    authorized_capabilities: Iterable[str] | None = None,
    registry: Mapping[str, CapabilityDefinition] | None = None,
) -> frozenset[str]:
    """Return tools whose capability is both image-verified and authorized."""
    available = frozenset(available_capabilities)
    authorized = available if authorized_capabilities is None else frozenset(authorized_capabilities)
    permitted = available & authorized
    definitions = CAPABILITIES if registry is None else registry
    tools: set[str] = set()
    for definition in definitions.values():
        if definition.requires <= permitted:
            tools.update(definition.tools)
    return frozenset(tools)
