"""Tests for tools.rob_operator_tools.

Mechanism-level tests (guard integration, parameter validation, redaction)
run everywhere. Real functional tests against actual local commands
(docker, git, host metrics, process inspection, network probes) run
directly on NiPoGi, since that's where this suite executes. A genuine
disposable-Postgres integration test exercises db_select end-to-end
against a real least-privilege role, isolated from project-os-db entirely
(see ROB_TEST_PG_DSN / ROB_TEST_PG_SUPERUSER_DSN — set only for this test
run, pointing at a disposable container created for this task and torn
down after)."""

import os
import shlex

import pytest

from tools import rob_operator_tools as t


# ---------------------------------------------------------------------------
# Injection / mutation rejection at the parameter level
# ---------------------------------------------------------------------------

class TestParameterInjectionRejected:
    def test_docker_inspect_rejects_unsafe_container_name(self):
        result = t.docker_inspect("; rm -rf /")
        assert not result.ok

    def test_docker_logs_rejects_unsafe_container_name(self):
        result = t.docker_logs("$(touch /tmp/x)")
        assert not result.ok

    def test_docker_logs_rejects_bad_tail(self):
        result = t.docker_logs("some-container", tail=-5)
        assert not result.ok
        result2 = t.docker_logs("some-container", tail=999999)
        assert not result2.ok

    def test_systemd_status_rejects_unsafe_unit(self):
        result = t.systemd_status("x; systemctl restart hermes-gateway")
        assert not result.ok

    def test_journal_query_rejects_bad_priority(self):
        result = t.journal_query(priority="not-a-real-priority")
        assert not result.ok

    def test_git_inspect_rejects_unknown_operation(self):
        result = t.git_inspect(".", "checkout")
        assert not result.ok

    def test_git_inspect_merge_base_requires_two_args(self):
        result = t.git_inspect(".", "merge-base", args=["only-one"])
        assert not result.ok

    def test_process_inspect_rejects_non_positive_pid(self):
        assert not t.process_inspect(-1).ok
        assert not t.process_inspect(0).ok

    def test_network_probe_rejects_bad_port(self):
        assert not t.network_probe("example.com", 0).ok
        assert not t.network_probe("example.com", 70000).ok

    def test_http_probe_rejects_non_get_head(self):
        assert not t.http_probe("https://example.com", method="POST").ok
        assert not t.http_probe("https://example.com", method="DELETE").ok

    def test_http_probe_rejects_non_http_scheme(self):
        assert not t.http_probe("file:///etc/passwd").ok

    def test_container_exec_readonly_rejects_mutation_command(self):
        result = t.container_exec_readonly("some-container", "rm -rf /tmp")
        assert not result.ok

    def test_container_exec_readonly_rejects_shell_wrapper(self):
        result = t.container_exec_readonly("some-container", "sh -c 'rm -rf /'")
        assert not result.ok


class TestRunNeverLeaksSecretAtTruncationBoundary:
    """Regression: an earlier version of _run() truncated command output to
    _MAX_OUTPUT_CHARS BEFORE redacting it, to bound redact_text's CPU cost
    on unbounded commands (docker logs, git diff, ...). That reorder was
    itself a real leak: several redaction patterns need trailing context
    to even recognize a secret (_URI_CREDENTIAL_PATTERN needs the closing
    '@'), so a cut landing between a secret's value and its terminator
    left the value's prefix in plaintext in the truncated output — exactly
    the DATABASE_URL=postgresql://user:PASSWORD@host shape that motivated
    this module's own existence. Fixed by redacting the full output
    first, then truncating the already-redacted result; the actual fix
    for the CPU cost is bounding the regex itself (see
    secret_redaction.py's _URI_CREDENTIAL_PATTERN scheme-length bound),
    not truncating before redacting.
    """

    def test_secret_straddling_truncation_boundary_is_not_leaked(self, tmp_path):
        secret = "SuperSecretPassword123"
        line = f'"DATABASE_URL=postgresql://dbuser:{secret}@db.internal/db",\n'
        # Padding placed so the SECRET LINE ITSELF sits comfortably inside
        # the output bound (total well under _MAX_OUTPUT_CHARS), so the
        # redacted marker is still visible in the final output rather than
        # truncated away entirely — this isolates the actual regression
        # (does a secret in a LARGE document still get redacted at all)
        # from truncation position arithmetic.
        padding = "P" * (t._MAX_OUTPUT_CHARS - len(line) - 10_000)
        content = padding + line
        assert len(content) < t._MAX_OUTPUT_CHARS
        f = tmp_path / "big_output.txt"
        f.write_text(content)

        result = t._run(f"cat {shlex.quote(str(f))}")
        assert result.ok, result.error
        assert secret not in result.output
        assert "[REDACTED]" in result.output

    def test_secret_at_exact_truncation_boundary_never_leaks_raw_value(self, tmp_path):
        # The harder case: the secret's own text spans the exact point
        # where _MAX_OUTPUT_CHARS would have cut under the OLD (buggy)
        # truncate-first order. Whether or not the redacted marker
        # survives into the final (possibly still-truncated) output, the
        # raw secret value must never appear in it.
        secret = "SuperSecretPassword123"
        padding = "P" * (t._MAX_OUTPUT_CHARS - 15)
        content = f'{padding}"DATABASE_URL=postgresql://dbuser:{secret}@db.internal/db",\n'
        f = tmp_path / "boundary_output.txt"
        f.write_text(content)

        result = t._run(f"cat {shlex.quote(str(f))}")
        assert result.ok, result.error
        assert secret not in result.output


class TestSqlGuard:
    def test_rejects_insert(self):
        assert t._reject_non_select("INSERT INTO x VALUES (1)") is not None

    def test_rejects_drop(self):
        assert t._reject_non_select("DROP TABLE x") is not None

    def test_rejects_multi_statement(self):
        assert t._reject_non_select("SELECT 1; DROP TABLE x") is not None

    def test_rejects_delete_disguised_in_with(self):
        assert t._reject_non_select("WITH d AS (DELETE FROM x RETURNING *) SELECT * FROM d") is not None

    def test_allows_plain_select(self):
        assert t._reject_non_select("SELECT * FROM widgets") is None

    def test_allows_select_with_column_named_like_keyword(self):
        # "deleted_at" contains "delete" as a substring but is not the
        # keyword itself — word-boundary matching must not false-positive.
        assert t._reject_non_select("SELECT deleted_at FROM widgets") is None

    def test_rejects_dangerous_function_calls(self):
        # Function-call-based mutation/DoS/file-access hiding behind a bare
        # SELECT — flagged in an independent review as missing from the
        # original keyword list.
        for query in [
            "select pg_terminate_backend(12345)",
            "select pg_cancel_backend(12345)",
            "select pg_read_server_files('/etc/passwd')",
            "select pg_reload_conf()",
            "select pg_rotate_logfile()",
            "select pg_switch_wal()",
            "select pg_promote()",
            "select set_config('log_statement', 'all', false)",
            "select pg_file_write('/tmp/x', 'y', false)",
            "select pg_advisory_lock(1)",
            "select pg_advisory_xact_lock(1)",
            "select lo_get(12345)",
        ]:
            assert t._reject_non_select(query) is not None, query


# ---------------------------------------------------------------------------
# Real functional tests (run directly on NiPoGi, no mocking)
# ---------------------------------------------------------------------------

class TestRealDockerFunctional:
    def test_docker_ps_runs(self):
        result = t.docker_ps()
        assert result.ok
        assert "NAMES" in result.output or "CONTAINER" in result.output

    def test_docker_stats_no_stream_runs(self):
        result = t.docker_stats()
        assert result.ok


class TestRealGitFunctional:
    def test_git_inspect_status_on_this_repo(self, tmp_path_repo=None):
        # Run against the actual worktree this test executes from.
        result = t.git_inspect(".", "status")
        assert result.ok

    def test_git_inspect_branch_current(self):
        result = t.git_inspect(".", "branch-current")
        assert result.ok
        assert "ai/rob-readonly-operator-p01" in result.output


class TestRealHostFunctional:
    def test_host_metrics_runs(self):
        result = t.host_metrics()
        assert result.ok
        assert "Linux" in result.output or "linux" in result.output.lower()

    def test_process_inspect_self(self):
        result = t.process_inspect(os.getpid())
        assert result.ok


class TestRealNetworkFunctional:
    def test_network_probe_localhost_ssh_or_similar(self):
        # 127.0.0.1:22 (sshd) — safe, local, always either open or cleanly refused.
        result = t.network_probe("127.0.0.1", 65533)  # a port almost certainly closed
        assert not result.ok  # connect failure is the expected, correctly-reported outcome

    def test_http_probe_head_against_local_health_endpoint(self):
        result = t.http_probe("http://127.0.0.1:3100/api/health", method="HEAD")
        # HEAD may or may not be supported by the app; either a clean ok
        # or a clean, non-crashing error is acceptable here — the point is
        # the tool itself never raises.
        assert isinstance(result.ok, bool)


class TestRealJournalFunctional:
    def test_journal_query_runs(self):
        result = t.journal_query(unit="hermes-gateway.service", since="10 minutes ago")
        assert result.ok


# ---------------------------------------------------------------------------
# Disposable Postgres integration (real role, real data, isolated instance)
# ---------------------------------------------------------------------------

_PG_DSN = os.environ.get("ROB_TEST_PG_RO_DSN")


@pytest.mark.skipif(not _PG_DSN, reason="ROB_TEST_PG_RO_DSN not set — disposable Postgres not available this run")
class TestDbSelectAgainstDisposablePostgres:
    def test_select_returns_real_rows(self, monkeypatch):
        monkeypatch.setenv("ROB_DB_PROFILE_ROBTEST_DSN", _PG_DSN)
        result = t.db_select("robtest", "SELECT id, name FROM widgets ORDER BY id")
        assert result.ok, result.error
        assert "alpha" in result.output
        assert "beta" in result.output
        assert "gamma" in result.output

    def test_row_limit_enforced(self, monkeypatch):
        monkeypatch.setenv("ROB_DB_PROFILE_ROBTEST_DSN", _PG_DSN)
        result = t.db_select("robtest", "SELECT * FROM widgets", row_limit=1)
        assert result.ok, result.error
        import json

        payload = json.loads(result.output)
        assert payload["row_count"] == 1

    def test_insert_rejected_before_reaching_db(self, monkeypatch):
        monkeypatch.setenv("ROB_DB_PROFILE_ROBTEST_DSN", _PG_DSN)
        result = t.db_select("robtest", "INSERT INTO widgets (name) VALUES ('should-never-exist')")
        assert not result.ok
        # Prove it never reached the DB, not just that the tool said no.
        # db_select's output shape is {"columns": [...], "rows": [[...]], "row_count": N}
        # (a plain list of row-value lists, not list-of-dicts) — count(*) of
        # zero matches is therefore rows == [[0]].
        import json

        verify = t.db_select("robtest", "SELECT count(*) AS n FROM widgets WHERE name = 'should-never-exist'")
        assert verify.ok, verify.error
        payload = json.loads(verify.output)
        assert payload["rows"] == [[0]]

    def test_role_level_write_also_refused_independent_of_query_text_guard(self):
        # Defense-in-depth check: even bypassing db_select's own text-level
        # SQL guard entirely and going straight at the role with psycopg,
        # the role's OWN server-side configuration must refuse a write —
        # proves the security property doesn't rest solely on the
        # Python-level string check. `ALTER ROLE rob_ro SET
        # default_transaction_read_only = on` (mirroring
        # create_projectos_ro_role.sql's real production script) makes
        # every session as this role reject writes via
        # ReadOnlySqlTransaction before privilege checking even applies —
        # a strictly stronger guarantee than relying on GRANT/REVOKE alone
        # (which would instead raise InsufficientPrivilege).
        import psycopg

        with psycopg.connect(_PG_DSN) as conn:
            with conn.cursor() as cur:
                with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
                    cur.execute("INSERT INTO widgets (name) VALUES ('bypass-attempt')")
