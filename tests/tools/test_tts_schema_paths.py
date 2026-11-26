"""Regression tests for hardcoded ~/.hermes in agent-facing schema strings.

PR #10285 replaced hardcoded ``~/.hermes`` references in tool schema
descriptions with ``display_hermes_home()`` so that custom HERMES_HOME
values (Docker, profiles) are reflected correctly in agent-visible text.

These tests guard against re-introducing literal ``~/.hermes`` in any
schema description string that the model sees, which would mislead the
agent into referencing the wrong directory when a profile or custom
HERMES_HOME is active.
"""

import re

import pytest


HARDCODED_PATTERN = re.compile(r"~/?\.hermes")


def _collect_schema_strings(obj):
    """Recursively yield all string values from a nested dict/list."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _collect_schema_strings(v)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _collect_schema_strings(item)


class TestTTSSchemaPath:
    """The TTS tool schema must not contain hardcoded ~/.hermes paths."""

    def test_output_path_description_uses_display_hermes_home(self):
        from tools.tts_tool import TTS_SCHEMA

        for text in _collect_schema_strings(TTS_SCHEMA):
            assert not HARDCODED_PATTERN.search(text), (
                f"TTS_SCHEMA still contains hardcoded ~/.hermes: {text!r}"
            )

    def test_output_path_description_contains_dynamic_home(self):
        from hermes_constants import display_hermes_home
        from tools.tts_tool import TTS_SCHEMA

        desc = TTS_SCHEMA["properties"]["output_path"]["description"]
        expected = display_hermes_home()
        assert expected in desc, (
            f"Expected '{expected}' in output_path description, got: {desc!r}"
        )


class TestCronjobSchemaPath:
    """Cronjob tool schema must use display_hermes_home()."""

    def test_script_description_no_hardcoded_path(self):
        from tools.cronjob_tools import CRONJOB_SCHEMA

        for text in _collect_schema_strings(CRONJOB_SCHEMA):
            assert not HARDCODED_PATTERN.search(text), (
                f"CRONJOB_SCHEMA still contains hardcoded ~/.hermes: {text!r}"
            )


class TestSkillManagerSchemaPath:
    """Skill manager schema must use display_hermes_home()."""

    def test_description_no_hardcoded_path(self):
        from tools.skill_manager_tool import SKILL_MANAGE_SCHEMA

        for text in _collect_schema_strings(SKILL_MANAGE_SCHEMA):
            assert not HARDCODED_PATTERN.search(text), (
                f"SKILL_MANAGE_SCHEMA still contains hardcoded ~/.hermes: {text!r}"
            )
