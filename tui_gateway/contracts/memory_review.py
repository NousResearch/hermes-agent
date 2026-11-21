"""Memory write review contracts."""
from typing import Literal
from .base import Params, Result
from .registry import method

class MemoryPendingParams(Params):
    session_id: str | None = None
    profile: str | None = None

class MemoryReviewBatch(Result):
    id: str
    summary: str
    origin: str
    created_at: float
    target: str
    action: str
    operation_count: int
    before: str
    after: str
    diff: str
    revision: str
    can_approve: bool
    error: str

class MemoryPendingResult(Result):
    write_approval: bool
    batches: list[MemoryReviewBatch]

class MemoryDecideParams(MemoryPendingParams):
    id: str
    decision: Literal["approve", "reject"]
    revision: str

class MemoryDecideResult(Result):
    success: bool
    error: str = ""

method("memory.pending", params=MemoryPendingParams, result=MemoryPendingResult)
method("memory.decide", params=MemoryDecideParams, result=MemoryDecideResult)
