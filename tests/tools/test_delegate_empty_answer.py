"""A child whose whole final answer is one empty JSON value returned no result.

Like run_agent's "(empty)" give-up sentinel, ``{}`` / ``[]`` / ``null`` / ``""`` carry nothing for the parent, so the
entry must not read ``status: completed``. A declared ``output_schema`` that accepts the value still makes it the
answer (an empty list of findings is a real result).
"""

import pytest

from tests.tools.test_delegate_output_schema import ADDRESS_SCHEMA, _StubChild, _run


@pytest.mark.parametrize(
    "answer, status",
    [
        ("{}", "failed"),
        ("[ ]", "failed"),
        ("null", "failed"),
        ('""', "failed"),
        ("```json\n{}\n```", "failed"),
        ("0", "completed"),
        ("false", "completed"),
        ('{"items": []}', "completed"),
        ("No findings: {} files matched.", "completed"),
    ],
)
def test_an_empty_json_answer_is_not_a_completed_result(answer, status):
    assert _run(_StubChild([answer]))["status"] == status


def test_the_declared_schema_decides_whether_an_empty_value_is_the_answer():
    """An empty array the schema accepts is the answer. An empty object the schema still rejects after the retry is
    not: the entry must not be completed with a note telling the parent to extract its result from ``{}`` rather
    than re-run the task."""
    accepted = _StubChild(["[]"])
    accepted._delegate_output_schema = {"type": "array", "items": {"type": "string"}}
    assert _run(accepted)["status"] == "completed"

    rejected = _StubChild(["{}", "{}"])
    rejected._delegate_output_schema = ADDRESS_SCHEMA
    entry = _run(rejected)
    assert entry["status"] == "failed"
    assert entry["schema_valid"] is False
    assert "schema_note" not in entry
