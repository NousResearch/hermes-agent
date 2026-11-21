"""Token-budget regression for structured delegation output contracts.

The realistic 80-metric schema follows dexhunter's benchmark for PR #154655.
"""

import json

from agent.model_metadata import estimate_tokens_rough
from tools.delegation_output_schema import append_output_contract


def test_wide_output_contract_preserves_schema_within_token_budget():
    properties = {}
    for i in range(80):
        properties[f"metric_{i}"] = {
            "type": "object",
            "description": (
                f"Metric {i} collected from the child task, including the unit, "
                "the window, and the caveats the parent must preserve."
            ),
            "properties": {
                "name": {"type": "string", "description": "Stable metric name."},
                "value": {"type": "number", "description": "Numeric value in the declared unit."},
                "unit": {"type": "string", "description": "Unit such as ms, bytes, or ratio."},
                "notes": {"type": "string", "description": "Caveats that must survive into the parent summary."},
            },
            "required": ["name", "value", "unit"],
        }
    schema = {
        "type": "object",
        "description": "Final structured report the child must emit as its only message.",
        "properties": properties,
        "required": [f"metric_{i}" for i in range(0, 80, 4)],
    }
    context = "Investigate the retry budget and return the structured report."

    prompt = append_output_contract(context, schema)
    assert prompt.startswith(context + "\n\nOUTPUT CONTRACT (machine-validated):\n")

    # Both indented and compact contracts must preserve the exact schema.
    schema_start = prompt.index("\n{", prompt.index("OUTPUT CONTRACT")) + 1
    assert json.loads(prompt[schema_start:]) == schema

    # The 80-metric contract must not spend the old ~15,685-token budget.
    # Allow headroom above the ~10,534-token compact result.
    assert estimate_tokens_rough(prompt) <= 12_000
