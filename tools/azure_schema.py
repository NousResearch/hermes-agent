"""Lossless tool-pattern preflight for Azure Foundry's regex validator.

Azure rejects Unicode property escapes such as RunPod's ``\\p{Z}`` before
executing the model. Expand the known, stable Unicode Separator category to
an equivalent BMP character class. Do NOT drop patterns or substitute ``\\s``:
that changes validation (e.g. tabs, NEL and zero-width space).

This guard is independent of the legacy, provider-agnostic schema sanitizer.
It runs at the request boundary, handles Chat Completions and Responses, and
leaves all non-pattern constraints unchanged. Unsupported/untranslatable
patterns fail locally, before network I/O, rather than weakening validation.
It is not a general ECMAScript-to-Python regex transpiler.
"""
from __future__ import annotations

import copy
from functools import lru_cache
import re
from typing import Any
from urllib.parse import urlsplit
import warnings


class AzureToolSchemaError(ValueError):
    """A tool schema cannot safely be sent to Azure's schema validator."""


# Unicode General_Category=Separator (Zs | Zl | Zp), 19 code points.
# Stable since Unicode 6.3; deliberately explicit, not locale/\s dependent.
_SEPARATOR_RANGES = r"\u0020\u00A0\u1680\u2000-\u200A\u2028\u2029\u202F\u205F\u3000"
_SCHEMA_MAPS = frozenset({"properties", "$defs", "definitions", "dependentSchemas"})
_SCHEMA_SINGLES = frozenset({
    "additionalProperties", "unevaluatedProperties", "additionalItems",
    "unevaluatedItems", "propertyNames", "contains", "not", "if", "then", "else",
})
_SCHEMA_ARRAYS = frozenset({"anyOf", "allOf", "oneOf", "prefixItems"})


def _invalid_pattern() -> AzureToolSchemaError:
    # Never echo schema text, defaults, examples, prompts, credentials or headers.
    return AzureToolSchemaError("Azure tool schema has an unsupported or malformed regular expression")


@lru_cache(maxsize=512)
def _portable_pattern(pattern: str) -> str:
    out: list[str] = []
    in_class = False
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if char == "\\":
            if i + 1 >= len(pattern):
                raise _invalid_pattern()
            escape = pattern[i + 1]
            if escape in {"p", "P"}:
                # Recognize only a proven equivalent, not arbitrary properties,
                # scripts, binary classes or ambiguous complement-in-class forms.
                if pattern[i + 2:i + 5] != "{Z}":
                    raise _invalid_pattern()
                end = i + 5
                if in_class and (escape == "P" or (out and out[-1] == "-") or pattern[end:end + 1] == "-"):
                    raise _invalid_pattern()
                if in_class:
                    out.append(_SEPARATOR_RANGES)
                else:
                    out.append("[" + ("^" if escape == "P" else "") + _SEPARATOR_RANGES + "]")
                i = end
                continue
            # Consume escaped pairs atomically: \\p is literal, \[ isn't a class.
            out.append(pattern[i:i + 2])
            i += 2
            continue
        if char == "[":
            # ECMAScript empty / negated-empty classes differ from Python re.
            # Do not silently reinterpret [] or [^] as a different class.
            if pattern.startswith("[]", i) or pattern.startswith("[^]", i):
                raise _invalid_pattern()
            in_class = True
        elif char == "]":
            in_class = False
        out.append(char)
        i += 1
    result = "".join(out)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            re.compile(result)
    except (re.error, FutureWarning, OverflowError, RecursionError):
        raise _invalid_pattern() from None
    return result


def _rewrite_pattern(value: Any) -> str:
    if not isinstance(value, str):
        raise _invalid_pattern()
    return _portable_pattern(value)


def _walk_schema(node: Any, state: set[str]) -> None:
    """Visit schema positions only; metadata and literal argument names are data."""
    if isinstance(node, bool):
        return
    if not isinstance(node, dict):
        raise AzureToolSchemaError("Azure tool parameters contain a non-schema value")
    if any(key in node for key in ("$ref", "$dynamicRef", "$recursiveRef")):
        state.add("reference")
    if "pattern" in node:
        node["pattern"] = _rewrite_pattern(node["pattern"])
    for key, value in list(node.items()):
        if key in _SCHEMA_MAPS or key == "patternProperties":
            if not isinstance(value, dict):
                raise AzureToolSchemaError("Azure tool schema has an invalid schema map")
            rewritten = {}
            for name, child in value.items():
                target = _rewrite_pattern(name) if key == "patternProperties" else name
                if target != name:
                    state.add("renamed")
                if target in rewritten:
                    raise AzureToolSchemaError("Azure tool schema pattern-property translation collides")
                _walk_schema(child, state)
                rewritten[target] = child
            node[key] = rewritten
        elif key in _SCHEMA_SINGLES:
            _walk_schema(value, state)
        elif key == "items" or key in _SCHEMA_ARRAYS:
            if isinstance(value, list):
                for child in value:
                    _walk_schema(child, state)
            elif key == "items":
                _walk_schema(value, state)
            else:
                raise AzureToolSchemaError("Azure tool schema has an invalid schema array")
        elif key == "dependencies" and isinstance(value, dict):
            # Draft-07 dependencies may be a schema OR a list of field names.
            for child in value.values():
                if isinstance(child, (dict, bool)):
                    _walk_schema(child, state)


def sanitize_azure_tool_schemas(tools: list[dict]) -> list[dict]:
    """Copy tool schemas; translate/validate regexes without losing constraints."""
    if not isinstance(tools, list):
        raise AzureToolSchemaError("Azure tools must be a list")
    result = copy.deepcopy(tools)
    for tool in result:
        if not isinstance(tool, dict):
            raise AzureToolSchemaError("Azure tool entry must be an object")
        if tool.get("type") != "function":
            continue  # Native web search, code interpreter, etc.
        fn = tool.get("function", tool)
        if not isinstance(fn, dict):
            raise AzureToolSchemaError("Azure function tool must be an object")
        if "parameters" in fn:
            state: set[str] = set()
            _walk_schema(fn["parameters"], state)
            # Renaming patternProperties may invalidate JSON Pointer references.
            # Do not guess through $id scopes or encoded pointers: fail locally.
            if state == {"reference", "renamed"}:
                raise AzureToolSchemaError("Azure tool schema combines references with renamed pattern properties")
    return result


def is_azure_schema_target(provider: str = "", base_url: Any = "") -> bool:
    """Provider identity first; hostname fallback for custom Azure endpoints."""
    if provider == "azure-foundry":
        return True
    try:
        host = (urlsplit(str(base_url or "")).hostname or "").lower()
    except ValueError:
        return False
    return any(host.endswith(suffix) for suffix in (
        ".openai.azure.com", ".services.ai.azure.com", ".cognitiveservices.azure.com",
        ".openai.azure.us", ".services.ai.azure.us", ".openai.azure.cn",
    ))


def prepare_azure_request(api_kwargs: dict, *, provider: str = "", base_url: Any = "") -> dict:
    """Preflight final SDK kwargs (including late extra_body.tools overrides).

    Only tool schemas are deep-copied. Model, messages, authentication objects,
    headers, timeout and all other request settings remain untouched.
    """
    if not is_azure_schema_target(provider, base_url):
        return api_kwargs
    result = dict(api_kwargs)
    def prepare_collection(value):
        if value is None:
            return value
        if isinstance(value, (list, tuple)):
            return sanitize_azure_tool_schemas(list(value))
        # Preserve SDK omission sentinels, but never let arbitrary iterables or
        # malformed collections bypass preflight and reach the network.
        from openai import NOT_GIVEN, omit
        if value is NOT_GIVEN or value is omit:
            return value
        raise AzureToolSchemaError("Azure tools must be a list or tuple")

    if "tools" in result:
        result["tools"] = prepare_collection(result["tools"])
    extra = result.get("extra_body")
    if isinstance(extra, dict) and "tools" in extra:
        result["extra_body"] = {**extra, "tools": prepare_collection(extra["tools"])}
    return result
