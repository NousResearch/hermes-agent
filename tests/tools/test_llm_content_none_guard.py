"""Test for extract_content_or_reasoning() guard against None/empty choices.

Issue: #5968 - TypeError or AttributeError when response.choices is None/empty
"""

import pytest
from unittest.mock import MagicMock


def test_choices_none_returns_empty():
    """When choices is None, should return empty string."""
    from agent.auxiliary_client import extract_content_or_reasoning

    response = MagicMock()
    response.choices = None

    result = extract_content_or_reasoning(response)
    assert result == ""


def test_choices_empty_returns_empty():
    """When choices is empty list, should return empty string."""
    from agent.auxiliary_client import extract_content_or_reasoning

    response = MagicMock()
    response.choices = []

    result = extract_content_or_reasoning(response)
    assert result == ""


def test_missing_choices_returns_empty():
    """When choices attribute is missing, should return empty string."""
    from agent.auxiliary_client import extract_content_or_reasoning

    response = MagicMock(spec=[])  # No attributes
    del response.choices  # Ensure choices is missing

    result = extract_content_or_reasoning(response)
    assert result == ""