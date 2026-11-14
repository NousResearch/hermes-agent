"""DELETE-without-WHERE approval rule: only a WHERE in the DELETE's own statement clears it.

Probe shapes from #118654 (with tests adapted from PR #118655 by beardthelion).
"""

import pytest

from tools.approval import detect_dangerous_command


@pytest.mark.parametrize("cmd", [
    'psql -c "DELETE FROM users -- WHERE"',
    'mysql -e "DELETE FROM t /* WHERE */"',
    'mysql -e "DELETE FROM t # WHERE"',
    'psql -c "DELETE FROM t /* WHERE"',
    "psql -c 'DELETE FROM t; SELECT 1 AS \"WHERE\"'",
    'psql -c "DELETE FROM t RETURNING \'WHERE\'"',
    'psql -c "DELETE FROM t RETURNING $$WHERE$$"',
    'psql -c "DELETE FROM t RETURNING $q$WHERE$q$"',
    "psql -c 'DELETE FROM t RETURNING $é$ WHERE $é$'",
    "psql -c 'DELETE FROM t RETURNING $日本$ WHERE $日本$'",
    'psql -c "DELETE FROM t" ; echo "WHERE"',
    'psql -c "DELETE FROM t" && echo WHERE',
    "bash -c \"psql -c 'DELETE FROM t'\"",
])
def test_where_outside_the_delete_statement_does_not_clear(cmd):
    is_dangerous, _, desc = detect_dangerous_command(cmd)
    assert is_dangerous is True, cmd
    assert desc == "SQL DELETE without WHERE"


@pytest.mark.parametrize("cmd", [
    'psql -c "DELETE FROM users WHERE id = 1"',
    "psql -c \"DELETE FROM t WHERE x = 'a;b'\"",
    'sqlite3 x.db "DELETE FROM t WHERE y = $$z$$"',
    'psql -c "DELETE FROM t USING u WHERE t.id = u.id"',
    'psql -c "DELETE FROM ${TABLE} WHERE id = 1"',
    "psql -c 'DELETE FROM t WHERE name = $é$x$é$'",
    "delete from t where x = 1",
])
def test_real_where_clause_still_clears(cmd):
    assert detect_dangerous_command(cmd) == (False, None, None), cmd


def test_where_comment_bypass_blocked_end_to_end(monkeypatch):
    from unittest.mock import patch

    from tools.approval import check_all_command_guards

    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    with patch("tools.approval_context._get_cron_approval_mode", return_value="deny"):
        blocked = check_all_command_guards('psql -c "DELETE FROM users -- WHERE"', "local")
        assert blocked["approved"] is False
        assert "DELETE" in (blocked.get("message") or "")
        assert check_all_command_guards('psql -c "DELETE FROM users WHERE id = 1"', "local")["approved"] is True
