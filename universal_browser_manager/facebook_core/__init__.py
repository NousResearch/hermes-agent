"""Shared storage and safety primitives for Facebook automation."""

from .application import FacebookApplicationService
from .repository import AmbiguousFriendError, FacebookRepository
from .schema import SCHEMA_VERSION, create_schema, ensure_schema
from .storage import canonical_db_path, connect

__all__ = [
    "AmbiguousFriendError",
    "FacebookApplicationService",
    "FacebookRepository",
    "SCHEMA_VERSION",
    "canonical_db_path",
    "connect",
    "create_schema",
    "ensure_schema",
]
