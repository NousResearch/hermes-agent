"""Process-local facts about binary MCP resources materialized by Hermes.

The carrier leaves model-facing strings/dicts unchanged. Server text and JSON
metadata cannot populate it; observers receive only artifacts actually cached
while rendering this result.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MCPResourceArtifact:
    server_name: str
    tool_name: str
    uri: str
    path: str
    mime_type: str
    size: int
    sha256: str


class _ArtifactText(str):
    def __new__(cls, value, artifacts):
        result = super().__new__(cls, value)
        result._artifacts = artifacts
        return result


class _ArtifactDict(dict):
    def __init__(self, value, artifacts):
        super().__init__(value)
        self._artifacts = artifacts


def get_artifacts(result) -> tuple[MCPResourceArtifact, ...]:
    """Read host-created facts, never similarly named fields in tool output."""
    return result._artifacts if isinstance(result, (_ArtifactText, _ArtifactDict)) else ()


def with_artifacts(result, artifacts):
    """Carry resource facts without adding fields to the result's JSON shape."""
    artifacts = tuple(artifacts)
    if not artifacts:
        return result
    return _ArtifactText(result, artifacts) if isinstance(result, str) else _ArtifactDict(result, artifacts)
