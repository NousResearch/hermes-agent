import io
import json
import logging
import os
import re
import typing
from contextlib import contextmanager
from textwrap import indent, wrap
from typing import Any, Dict, Generator, Iterator, List, Optional, Sequence, Union

from .fastjsonschema_exceptions import JsonSchemaValueException

if typing.TYPE_CHECKING:
    import sys

    if sys.version_info < (3, 11):
        from typing_extensions import Self
    else:
        from typing import Self

_logger = logging.getLogger(__name__)

_MESSAGE_REPLACEMENTS = {
    "must be named by propertyName definition": "keys must be named by",
    "one of contains definition": "at least one item that matches",
    " same as const definition:": "",
    "only specified items": "only items matching the definition",
}

_SKIP_DETAILS = (
    "must not be empty",
    "is always invalid",
    "must not be there",
)

_NEED_DETAILS = {"anyOf", "oneOf", "allOf", "contains", "propertyNames", "not", "items"}

_CAMEL_CASE_SPLITTER = re.compile(r"\W+|([A-Z][^A-Z\W]*)")
_IDENTIFIER = re.compile(r"^[\w_]+$", re.I)

_TOML_JARGON = {
    "object": "table",
    "property": "key",
    "properties": "keys",
    "property names": "keys",
}

_FORMATS_HELP = """
For more details about `format` see
https://validate-pyproject.readthedocs.io/en/latest/api/validate_pyproject.formats.html
"""


class ValidationError(JsonSchemaValueException):
    """Report violations of a given JSON schema.

    This class extends :exc:`~fastjsonschema.JsonSchemaValueException`
    by adding the following properties:

    - ``summary``: an improved version of the ``JsonSchemaValueException`` error message
      with only the necessary information)

    - ``details``: more contextual information about the error like the failing schema
      itself and the value that violates the schema.

    Depending on the level of the verbosity of the ``logging`` configuration
    the exception message will be only ``summary`` (default) or a combination of
    ``summary`` and ``details`` (when the logging level is set to :obj:`logging.DEBUG`).
    """

    summary = ""
    details = ""
    _original_message = ""

    @classmethod
    def _from_jsonschema(cls, ex: JsonSchemaValueException) -> "Self":
        formatter = _ErrorFormatting(ex)
        obj = cls(str(formatter), ex.value, formatter.name, ex.definition, ex.rule)
        debug_code = os.getenv("JSONSCHEMA_DEBUG_CODE_GENERATION", "false").lower()
        if debug_code != "false":  # pragma: no cover
            obj.__cause__, obj.__traceback__ = ex.__cause__, ex.__traceback__
        obj._original_message = ex.message
        obj.summary = formatter.summary
        obj.details = formatter.details
        return obj


@contextmanager
def detailed_errors() -> Generator[None, None, None]:
    try:
        yield
    except JsonSchemaValueException as ex:
        raise ValidationError._from_jsonschema(ex) from None


class _ErrorFormatting:
    def __init__(self, ex: JsonSchemaValueException):
        self.ex = ex
        self.name = f"`{self._simplify_name(ex.name)}`"
        self._original_message: str = self.ex.message.replace(ex.name, self.name)
        self._summary = ""
        self._details = ""

    def __str__(self) -> str:
        if _logger.getEffectiveLevel() <= logging.DEBUG and self.details:
            return f"{self.summary}\n\n{self.details}"

        return self.summary

    @property
    def summary(self) -> str:
        if not self._summary:
            self._summary = self._expand_summary()

        return self._summary

    @property
    def details(self) -> str:
        if not self._details:
            self._details = self._expand_details()

        return self._details

    @staticmethod
    def _simplify_name(name: str) -> str:
        x = len("data.")
        return name[x:] if name.startswith("data.") else name

    def _expand_summary(self) -> str:
        msg = self._original_message

        for bad, repl in _MESSAGE_REPLACEMENTS.items():
            msg = msg.replace(bad, repl)

        if any(substring in msg for substring in _SKIP_DETAILS):
            return msg

        schema = self.ex.rule_definition
        if self.ex.rule in _NEED_DETAILS and schema:
            summary = _SummaryWriter(_TOML_JARGON)
            return f"{msg}:\n\n{indent(summary(schema), '    ')}"

        return msg

    def _expand_details(self) -> str:
        optional = []
        definition = self.ex.definition or {}
        desc_lines = definition.pop("$$description", [])
        desc = definition.pop("description", None) or " ".join(desc_lines)
        if desc:
            description = "\n".join(
                wrap(
                    desc,
                    width=80,
                    initial_indent="    ",
                    subsequent_indent="    ",
                    break_long_words=False,
                )
            )
            optional.append(f"DESCRIPTION:\n{description}")
        schema = json.dumps(definition, indent=4)
        value = json.dumps(self.ex.value, indent=4)
        defaults = [
            f"GIVEN VALUE:\n{indent(value, '    ')}",
            f"OFFENDING RULE: {self.ex.rule!r}",
            f"DEFINITION:\n{indent(schema, '    ')}",
        ]
        msg = "\n\n".join(optional + defaults)
        epilog = f"\n{_FORMATS_HELP}" if "format" in msg.lower() else ""
        return msg + epilog


class _SummaryWriter:
    _IGNORE = frozenset(("description", "default", "title", "examples"))

    def __init__(self, jargon: Optional[Dict[str, str]] = None):
        self.jargon: Dict[str, str] = jargon or {}
        # Clarify confusing terms
        self._terms = {
            "anyOf": "at least one of the following",
            "oneOf": "exactly one of the following",
            "allOf": "all of the following",
            "not": "(*NOT* the following)",
            "prefixItems": f"{self._jargon('items')} (in order)",
            "items": "items",
            "contains": "contains at least one of",
            "propertyNames": (
                f"non-predefined acceptable {self._jargon('property names')}"
            ),
            "patternProperties": f"{self._jargon('properties')} named via pattern",
            "const": "predefined value",
            "enum": "one of",
        }
        # Attributes that indicate that the definition is easy and can be done
        # inline (e.g. string and number)
        self._guess_inline_defs = [
            "enum",
            "const",
            "maxLength",
            "minLength",
            "pattern",
            "format",
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
        ]

    def _jargon(self, term: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(term, list):
            return [self.jargon.get(t, t) for t in term]
        return self.jargon.get(term, term)

    def __call__(
        self,
        schema: Union[dict, List[dict]],
        prefix: str = "",
        *,
        _path: Sequence[str] = (),
    ) -> str:
        if isinstance(schema, list):
            return self._handle_list(schema, prefix, _path)

        filtered = self._filter_unecessary(schema, _path)
        simple = self._handle_simple_dict(filtered, _path)
        if simple:
            return f"{prefix}{simple}"

        child_prefix = self._child_prefix(prefix, "  ")
        item_prefix = self._child_prefix(prefix, "- ")
        indent = len(prefix) * " "
        with io.StringIO() as buffer:
            for i, (key, value) in enumerate(filtered.items()):
                child_path = [*_path, key]
                line_prefix = prefix if i == 0 else indent
                buffer.write(f"{line_prefix}{self._label(child_path)}:")
                # ^  just the first item should receive the complete prefix
                if isinstance(value, dict):
                    filtered = self._filter_unecessary(value, child_path)
                    simple = self._handle_simple_dict(filtered, child_path)
                    buffer.write(
                        f" {simple}"
                        if simple
                        else f"\n{self(value, child_prefix, _path=child_path)}"
                    )
                elif isinstance(value, list) and (
                    key != "type" or self._is_property(child_path)
                ):
                    children = self._handle_list(value, item_prefix, child_path)
                    sep = " " if children.startswith("[") else "\n"
                    buffer.write(f"{sep}{children}")
                else:
                    buffer.write(f" {self._value(value, child_path)}\n")
            return buffer.getvalue()

    def _is_unecessary(self, path: Sequence[str]) -> bool:
        if self._is_property(path) or not path:  # empty path => instruction @ root
            return False
        key = path[-1]
        return any(key.startswith(k) for k in "$_") or key in self._IGNORE

    def _filter_unecessary(
        self, schema: Dict[str, Any], path: Sequence[str]
    ) -> Dict[str, Any]:
        return {
            key: value
            for key, value in schema.items()
            if not self._is_unecessary([*path, key])
        }

    def _handle_simple_dict(self, value: dict, path: Sequence[str]) -> Optional[str]:
        inline = any(p in value for p in self._guess_inline_defs)
        simple = not any(isinstance(v, (list, dict)) for v in value.values())
        if inline or simple:
            return f"{{{', '.join(self._inline_attrs(value, path))}}}\n"
        return None

    def _handle_list(
        self, schemas: list, prefix: str = "", path: Sequence[str] = ()
    ) -> str:
        if self._is_unecessary(path):
            return ""

        repr_ = repr(schemas)
        if all(not isinstance(e, (dict, list)) for e in schemas) and len(repr_) < 60:
            return f"{repr_}\n"

        item_prefix = self._child_prefix(prefix, "- ")
        return "".join(
            self(v, item_prefix, _path=[*path, f"[{i}]"]) for i, v in enumerate(schemas)
        )

    def _is_property(self, path: Sequence[str]) -> bool:
        """Check if the given path can correspond to an arbitrarily named property"""
        counter = 0
        for key in path[-2::-1]:
            if key not in {"properties", "patternProperties"}:
                break
            counter += 1

        # If the counter if even, the path correspond to a JSON Schema keyword
        # otherwise it can be any arbitrary string naming a property
        return counter % 2 == 1

    def _label(self, path: Sequence[str]) -> str:
        *parents, key = path
        if not self._is_property(path):
            norm_key = _separate_terms(key)
            return self._terms.get(key) or " ".join(self._jargon(norm_key))

        if parents[-1] == "patternProperties":
            return f"(regex {key!r})"
        return repr(key)  # property name

    def _value(self, value: Any, path: Sequence[str]) -> str:
        if path[-1] == "type" and not self._is_property(path):
            type_ = self._jargon(value)
            return f"[{', '.join(type_)}]" if isinstance(type_, list) else type_
        return repr(value)

    def _inline_attrs(self, schema: dict, path: Sequence[str]) -> Iterator[str]:
        for key, value in schema.items():
            child_path = [*path, key]
            yield f"{self._label(child_path)}: {self._value(value, child_path)}"

    def _child_prefix(self, parent_prefix: str, child_prefix: str) -> str:
        return len(parent_prefix) * " " + child_prefix


def _separate_terms(word: str) -> List[str]:
    """
    >>> _separate_terms("FooBar-foo")
    ['foo', 'bar', 'foo']
    """
    return [w.lower() for w in _CAMEL_CASE_SPLITTER.split(word) if w]
