from functools import reduce
from typing import Any, Callable, Dict

from . import formats
from .error_reporting import detailed_errors, ValidationError
from .extra_validations import EXTRA_VALIDATIONS
from .fastjsonschema_exceptions import JsonSchemaException, JsonSchemaValueException
from .fastjsonschema_validations import validate as _validate

__all__ = [
    "validate",
    "FORMAT_FUNCTIONS",
    "EXTRA_VALIDATIONS",
    "ValidationError",
    "JsonSchemaException",
    "JsonSchemaValueException",
]


FORMAT_FUNCTIONS: Dict[str, Callable[[str], bool]] = {
    fn.__name__.replace("_", "-"): fn
    for fn in formats.__dict__.values()
    if callable(fn) and not fn.__name__.startswith("_")
}


def validate(data: Any) -> bool:
    """Validate the given ``data`` object using JSON Schema
    This function raises ``ValidationError`` if ``data`` is invalid.
    """
    with detailed_errors():
        _validate(data, custom_formats=FORMAT_FUNCTIONS)
        reduce(lambda acc, fn: fn(acc), EXTRA_VALIDATIONS, data)
    return True
