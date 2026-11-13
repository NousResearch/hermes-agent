"""Real guard regressions for shell comments in the #117827 review.

Only command strings reach the approval guard; no shell commands are executed.
"""

from unittest.mock import Mock

import pytest

import hermes_cli.config as config
from agent import auxiliary_client
from tools import approval


@pytest.fixture
def smart_guard(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "approvals:\n  mode: smart\n"
        "command_allowlist: []\n"
        "security:\n  tirith_enabled: false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.delenv("TIRITH_ENABLED", raising=False)
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    config._LOAD_CONFIG_CACHE.clear()
    guardian = Mock(side_effect=AssertionError("The model must not decide hardline or inert-comment cases"))
    monkeypatch.setattr(auxiliary_client, "call_llm", guardian)
    deny = Mock(return_value="deny")
    yield deny, guardian
    config._LOAD_CONFIG_CACHE.clear()


@pytest.mark.parametrize("command", [
    "printf SAFE\r#x; reboot",
    "printf SAFE\u00a0#x; reboot",
    "printf SAFE\v#x; reboot",
    "printf SAFE\f#x; reboot",
    "printf SAFE\u2003#x; reboot",
    "printf SAFE\\ #x; reboot",
    "printf SAFE\\\t#x; reboot",
    "printf SAFE\\;#x; reboot",
    "printf SAFE\\\n#x; reboot",
    "printf 'SAFE '#x; reboot",
    'printf "SAFE "#x; reboot',
    "printf $(printf hi)#x; reboot",
    "printf $((1 + 2))#x; reboot",
    "printf ${value:-SAFE}#x; reboot",
    "printf `printf hi`#x; reboot",
    "printf <(printf hi)#x; reboot",
    "printf @(SAFE)#x; reboot",
    "values=(SAFE)#x; reboot",
    "printf '%s' $( (printf hi))#x; reboot",
    "printf '%s' $(case x in x) printf hi;; esac)#x; reboot",
    "printf '%s' $(printf '%s' then case x in x)#; reboot",
    "printf '%s' $(printf '%s' ${value:- # literal}); reboot",
    "printf '%s' $(printf '%s' ${value:- # literal})#x; reboot",
    "[[ a =~ (a)# ]]; reboot",
    "printf '%s' $\\\n'\\' #'; reboot",
    "(printf SAFE)# comment\nreboot",
    "safe()# comment\n{ printf SAFE; }; reboot",
])
def test_executable_suffix_is_hardline_blocked_without_forcing_a_warning(command, smart_guard):
    deny, guardian = smart_guard
    result = approval.check_all_command_guards(command, "local", approval_callback=deny)

    assert result["approved"] is False
    assert "hardline" in result["message"].lower()
    deny.assert_not_called()
    guardian.assert_not_called()


@pytest.mark.parametrize("command", [
    "printf SAFE # comment; reboot",
    "printf SAFE\t# comment; reboot",
    "printf SAFE;# comment; reboot",
    "printf SAFE|# comment; reboot\ncat",
    "printf SAFE&# comment; reboot",
    "(printf SAFE)# comment; reboot",
    "((1 + 2))# comment; reboot",
    "safe()# comment; reboot\n{ printf SAFE; }; safe",
    "case x in x)# comment; reboot\nprintf SAFE;; esac",
    "printf SAFE \\\n# comment; reboot",
    "(printf SAFE)\\\n# comment; reboot",
    "printf '%s' <((printf SAFE)# comment; reboot\n)#x",
    "printf '%s' $\\\n'it\\'s' # comment; reboot",
    "printf $(printf hi) # comment; reboot",
    "printf $((1 + 2)) # comment; reboot",
    'printf "$(printf hi)" # comment; reboot',
])
def test_genuine_comments_do_not_trigger_approval(command, smart_guard):
    deny, guardian = smart_guard
    result = approval.check_all_command_guards(command, "local", approval_callback=deny)

    assert result["approved"] is True
    deny.assert_not_called()
    guardian.assert_not_called()
