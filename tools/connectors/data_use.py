"""Data-use requirements of tool results, collected while one tool call dispatches.

The executor binds a sink around each dispatch; connector dispatch and plugin tools
(``PluginContext.record_data_use``) add to it, so a batch's entries union into the wrapper's sink and
the requirement never rides the model-visible result string.
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import Iterable, Iterator, Literal, Optional, get_args

DataUseRequirement = Literal["no_training", "zero_retention"]
DATA_USE_REQUIREMENTS: frozenset[str] = frozenset(get_args(DataUseRequirement))

_sink: contextvars.ContextVar[Optional[set[str]]] = contextvars.ContextVar("connector_data_use", default=None)


def parse_data_use_requirements(requires: Iterable[str]) -> tuple[str, ...]:
    """*requires* as sorted, de-duplicated names from :data:`DATA_USE_REQUIREMENTS`; ``ValueError`` otherwise.

    A bare ``str`` is refused, not iterated: ``"no_training"`` would read as eleven one-letter names."""
    if isinstance(requires, (str, bytes)):
        raise ValueError(f"data-use requirements must be a list of names, not the single value {requires!r}")
    try:
        items = list(requires)
    except TypeError:
        raise ValueError(f"data-use requirements must be a list of names, not {type(requires).__name__}") from None
    unknown = [item for item in items if not isinstance(item, str) or item not in DATA_USE_REQUIREMENTS]
    if unknown:
        raise ValueError(f"unknown data-use requirement(s) {unknown!r}; expected any of {sorted(DATA_USE_REQUIREMENTS)}")
    return tuple(sorted(set(items)))


@contextlib.contextmanager
def collect_data_use() -> Iterator[set[str]]:
    sink: set[str] = set()
    token = _sink.set(sink)
    try:
        yield sink
    finally:
        _sink.reset(token)


def record_data_use(requires: Iterable[str]) -> None:
    """No-op outside an agent dispatch (execute_code RPC, MCP serve): no message is built there."""
    sink = _sink.get()
    if sink is not None:
        sink.update(requires)
