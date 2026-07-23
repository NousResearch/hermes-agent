from tools.environments.local import LocalEnvironment


def _read_budget(env: LocalEnvironment) -> str:
    result = env.execute(
        "printf '%s/%s' "
        '"$HERMES_ITERATIONS_REMAINING" "$HERMES_MAX_ITERATIONS"'
    )
    assert result["returncode"] == 0
    return result["output"].strip()


def test_kanban_budget_refreshes_after_snapshot(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_budget")
    monkeypatch.setenv("HERMES_ITERATIONS_REMAINING", "89")
    monkeypatch.setenv("HERMES_MAX_ITERATIONS", "90")
    env = LocalEnvironment(
        cwd=str(tmp_path),
        timeout=10,
        env={
            "HERMES_ITERATIONS_REMAINING": "89",
            "HERMES_MAX_ITERATIONS": "90",
        },
    )
    try:
        assert _read_budget(env) == "89/90"

        monkeypatch.setenv("HERMES_ITERATIONS_REMAINING", "14")
        assert _read_budget(env) == "14/90"

        inline = env.execute(
            "HERMES_ITERATIONS_REMAINING=90 "
            "/bin/bash -c 'printf \"inline=%s/%s\" "
            "\"$HERMES_ITERATIONS_REMAINING\" \"$HERMES_MAX_ITERATIONS\"'"
        )
        assert "inline=90/90" not in inline["output"]
        assert "inline=14/90" in inline["output"]

        override = env.execute(
            "export HERMES_ITERATIONS_REMAINING=90; "
            "printf '%s/%s' "
            '"$HERMES_ITERATIONS_REMAINING" "$HERMES_MAX_ITERATIONS"'
        )
        assert "90/90" not in override["output"]
        assert "14/90" in override["output"]

        monkeypatch.setenv("HERMES_ITERATIONS_REMAINING", "13")
        assert _read_budget(env) == "13/90"
    finally:
        env.cleanup()
