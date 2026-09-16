"""Regression tests: _format_connect_error must survive __cause__/__context__ cycles.

The original recursive _find_missing/_flatten_messages walked children with no
visited-set; a self-referential exception chain (transport error whose __cause__
has __context__ pointing back at it — mcp-remote produces these) exhausted the
recursion limit and crashed `hermes mcp add/test` instead of printing the error.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.mcp_tool_errors import _format_connect_error


def _cyclic_chain():
    """ConnectionError <- ValueError (cause), ValueError.__context__ -> the ConnectionError."""
    try:
        try:
            raise RuntimeError("inner timeout")
        except RuntimeError as deepest:
            raise ValueError("middle layer") from deepest
    except ValueError as inner:
        outer = ConnectionError("stdio transport failed")
        outer.__cause__ = inner
        inner.__context__ = outer  # cycle
        return outer


def test_cyclic_cause_context_chain_does_not_recurse_to_death():
    result = _format_connect_error(_cyclic_chain())
    assert "stdio transport failed" in result
    # All three distinct messages survive the walk despite the cycle.
    assert "middle layer" in result
    assert "inner timeout" in result


def test_self_referencing_exception():
    exc = TimeoutError("self loop")
    exc.__context__ = exc  # degenerate single-node cycle
    result = _format_connect_error(exc)
    assert "self loop" in result


def test_file_not_found_still_extracted_under_cycle():
    fnf = FileNotFoundError(2, "No such file or directory", "C:/fake/npx.cmd")
    wrapper = ConnectionError("wrapped")
    wrapper.__cause__ = fnf
    fnf.__context__ = wrapper  # cycle around the FNF node
    result = _format_connect_error(wrapper)
    assert "missing executable" in result
    assert "C:/fake/npx.cmd" in result


def test_exception_group_children_flattened():
    group = ExceptionGroup("grp", [RuntimeError("alpha"), TimeoutError("beta")])
    result = _format_connect_error(group)
    assert "alpha" in result and "beta" in result
