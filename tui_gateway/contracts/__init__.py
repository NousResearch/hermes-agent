"""Wire contracts (see ``base.py``). Importing this package fills the registry tables; every topic
module is listed here so the generator and the runtime see the same catalog."""

from . import common, liveness, server_requests  # noqa: F401
from .base import JsonValue, Params, Payload, Result, WireEnum
from .registry import EVENTS, METHODS, SERVER_REQUESTS

__all__ = ["EVENTS", "METHODS", "SERVER_REQUESTS", "JsonValue", "Params", "Payload", "Result", "WireEnum"]
