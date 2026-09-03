"""Subagents must not inherit a reasoning config their model cannot honor.

Regression: with ``delegation.reasoning_effort`` unset or empty, the override
gate in delegate_tool was skipped entirely and the child inherited the parent's
reasoning config. A frontier parent with thinking enabled then handed
``thinking`` to a local worker model that does not support it, and Ollama
answered ``HTTP 400: "qwen3-coder-next" does not support thinking`` — a
non-retryable client error that killed the whole delegation batch.

Inheriting only makes sense when the child runs the parent's model. When
delegation routes to a different model (the normal case — a cheap local
worker), the parent's reasoning settings say nothing about what the child
supports, so they must not be carried over silently.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.delegate_tool import _resolve_child_reasoning

PARENT = {"enabled": True, "effort": "high"}


def test_unset_effort_does_not_inherit_thinking_for_a_different_child_model():
    """The bug: empty string skipped the gate and inherited the parent."""
    child = _resolve_child_reasoning(
        delegation_cfg={"reasoning_effort": ""},
        parent_reasoning=PARENT,
        parent_model="gpt-5.6-terra",
        child_model="qwen3-coder-next",
    )
    assert child == {"enabled": False}


def test_missing_effort_key_does_not_inherit_thinking_for_a_different_child_model():
    child = _resolve_child_reasoning(
        delegation_cfg={},
        parent_reasoning=PARENT,
        parent_model="gpt-5.6-terra",
        child_model="qwen3-coder-next",
    )
    assert child == {"enabled": False}


def test_unset_effort_still_inherits_when_child_runs_the_parent_model():
    """Same model = the parent's settings are known-good for the child."""
    child = _resolve_child_reasoning(
        delegation_cfg={"reasoning_effort": ""},
        parent_reasoning=PARENT,
        parent_model="gpt-5.6-terra",
        child_model="gpt-5.6-terra",
    )
    assert child == PARENT


def test_explicit_effort_is_honored_even_for_a_different_child_model():
    """An operator who asks for thinking on a capable worker still gets it."""
    child = _resolve_child_reasoning(
        delegation_cfg={"reasoning_effort": "medium"},
        parent_reasoning=PARENT,
        parent_model="gpt-5.6-terra",
        child_model="gpt-oss:120b",
    )
    assert child == {"enabled": True, "effort": "medium"}


def test_explicit_none_disables_thinking():
    child = _resolve_child_reasoning(
        delegation_cfg={"reasoning_effort": "none"},
        parent_reasoning=PARENT,
        parent_model="gpt-5.6-terra",
        child_model="gpt-oss:120b",
    )
    assert child == {"enabled": False}


def test_yaml_boolean_false_disables_thinking():
    """``reasoning_effort: false`` in YAML arrives as a bool, not a string."""
    child = _resolve_child_reasoning(
        delegation_cfg={"reasoning_effort": False},
        parent_reasoning=PARENT,
        parent_model="gpt-5.6-terra",
        child_model="gpt-oss:120b",
    )
    assert child == {"enabled": False}


def test_unrecognized_effort_falls_back_to_not_inheriting():
    """A typo must not silently re-enable the failure mode."""
    child = _resolve_child_reasoning(
        delegation_cfg={"reasoning_effort": "sky-high"},
        parent_reasoning=PARENT,
        parent_model="gpt-5.6-terra",
        child_model="qwen3-coder-next",
    )
    assert child == {"enabled": False}


def test_parent_without_reasoning_stays_without_reasoning():
    child = _resolve_child_reasoning(
        delegation_cfg={},
        parent_reasoning=None,
        parent_model="gpt-5.6-terra",
        child_model="gpt-5.6-terra",
    )
    assert child is None


if __name__ == "__main__":  # pytest is not installed in the Hermes venv
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{failures} failing" if failures else "\nall green")
    sys.exit(1 if failures else 0)
