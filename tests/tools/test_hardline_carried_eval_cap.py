"""An eval chain past the nesting cap fails closed inside a carried payload too (#155348 x #155258)."""
import pytest

from tools.approval_detection import _PARSER_LIMIT_DESCRIPTION, detect_hardline_command

_DEEP_EVAL = "eval " * 12 + "ls"


@pytest.mark.parametrize("command", [
    _DEEP_EVAL,
    "su -c '" + _DEEP_EVAL + "'",
    "env -S '" + _DEEP_EVAL + "'",
    "nice " + _DEEP_EVAL,
])
def test_over_nested_eval_refuses_in_every_carrier(command):
    assert detect_hardline_command(command) == (True, _PARSER_LIMIT_DESCRIPTION)
