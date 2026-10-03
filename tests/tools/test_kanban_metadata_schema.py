"""Free-form ``metadata`` objects on Kanban lifecycle tools must stay open after
schema sanitization. The sanitizer fills a bare ``{"type": "object"}`` with
``properties: {}`` / ``required: []``; without ``additionalProperties: true``
that reads as "an object with no fields", and grammar-constrained models emit
only ``{}`` — dropping required keys such as ``metadata.published_pr`` so PR
acceptance refuses every completion.
"""
from __future__ import annotations

import pytest

from tools import kanban_tools_schemas as schemas
from tools.schema_sanitizer import sanitize_tool_schemas

_FREE_FORM_METADATA_TOOLS = [
    schemas.KANBAN_COMPLETE_SCHEMA,
    schemas.KANBAN_REQUEST_REVIEW_SCHEMA,
]


@pytest.mark.parametrize("schema", _FREE_FORM_METADATA_TOOLS, ids=lambda s: s["name"])
def test_metadata_accepts_arbitrary_keys_after_sanitization(schema):
    (tool,) = sanitize_tool_schemas([{"type": "function", "function": schema}])
    metadata = tool["function"]["parameters"]["properties"]["metadata"]
    assert metadata["type"] == "object"
    assert metadata.get("additionalProperties") is True
