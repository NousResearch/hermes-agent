from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RepoRecord:
    id: int
    root: str
    root_hash: str
    created_at: float
    updated_at: float


@dataclass(frozen=True)
class FileRecord:
    id: int
    repo_id: int
    path: str
    language: str
    size: int
    mtime_ns: int
    sha256: str
    indexed_at: float


@dataclass(frozen=True)
class Symbol:
    id: int
    repo_id: int
    file_id: int
    name: str
    qualname: str
    kind: str
    start_line: int
    end_line: int
    signature: str | None = None
    docstring: str | None = None


@dataclass(frozen=True)
class Edge:
    id: int
    repo_id: int
    source_kind: str
    source_id: int
    target_kind: str
    target_id: int
    edge_type: str
    metadata_json: str = "{}"


@dataclass(frozen=True)
class ChunkRecord:
    id: int
    repo_id: int
    file_id: int
    start_line: int
    end_line: int
    text: str


@dataclass(frozen=True)
class IndexRun:
    id: int
    repo_id: int
    started_at: float
    finished_at: float | None
    files_seen: int
    files_indexed: int
    status: str
    message: str | None = None

