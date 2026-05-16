#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable


SCHEMA = "hermes.autonomy.crypto_bot_gitea_runner_recovery.v1"
RUNNER_NAME = "crypto-bot-linux-runner"
GITEA_CONTAINER = "crypto-bot-gitea"
RUNNER_IMAGE = "gitea/act_runner:0.2.12"
RUNNER_NETWORK = "crypto-bot-gitea-net"
RUNNER_VOLUME = "crypto-bot-linux-runner-data"
RUNNER_STATE_FILE = "/data/.runner"
RUNNER_DAEMON_COMMAND = "act_runner daemon"
INSTANCE_URL = "http://crypto-bot-gitea:3000"
RUNNER_LABELS = "linux,crypto-bot-python-313,ubuntu-latest"
REPO_SCOPE = "preston/crypto_bot"
GITEA_CONFIG_PATH = "/data/gitea/conf/app.ini"
GITEA_WORK_PATH = "/data/gitea"
APPROVAL_PHRASE = (
    "APPROVE CRYPTO_BOT GITEA RUNNER RECOVERY "
    "container=crypto-bot-linux-runner "
    "network=crypto-bot-gitea-net "
    "labels=linux,crypto-bot-python-313,ubuntu-latest "
    "no_workflow_dispatch=true "
    "no_pr_mutation=true "
    "no_merge=true"
)

CommandRunner = Callable[[list[str], int], dict[str, Any]]
SleepFn = Callable[[float], None]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def run_command(argv: list[str], timeout: int = 30) -> dict[str, Any]:
    proc = subprocess.run(
        argv,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
    )
    return {
        "argv": argv,
        "exit_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def command_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "exit_code": result.get("exit_code"),
        "stderr_tail": str(result.get("stderr") or "")[-500:],
    }


def docker_inspect_exists(
    name: str,
    *,
    runner: CommandRunner = run_command,
) -> bool:
    result = runner(["docker", "container", "inspect", name], 20)
    return result["exit_code"] == 0


def inspect_runner(
    *,
    runner_name: str = RUNNER_NAME,
    runner: CommandRunner = run_command,
) -> dict[str, Any]:
    ps = runner(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"name=^{runner_name}$",
            "--format",
            "{{.Names}}\t{{.Status}}\t{{.Image}}",
        ],
        20,
    )
    logs = runner(["docker", "logs", "--tail", "80", runner_name], 20)
    log_text = str(logs.get("stdout") or "") + str(logs.get("stderr") or "")
    return {
        "container": runner_name,
        "docker_ps": str(ps.get("stdout") or "").strip(),
        "docker_ps_exit_code": ps.get("exit_code"),
        "docker_logs_exit_code": logs.get("exit_code"),
        "token_empty_loop_detected": "token is empty" in log_text,
        "instance_empty_loop_detected": "instance address is empty" in log_text,
        "registered_successfully_detected": "Runner registered successfully" in log_text,
        "recent_log_tail": log_text[-2000:],
    }


def generate_registration_token(
    *,
    runner: CommandRunner = run_command,
) -> tuple[str | None, dict[str, Any]]:
    result = runner(
        [
            "docker",
            "exec",
            "--user",
            "git",
            GITEA_CONTAINER,
            "/usr/local/bin/gitea",
            "--config",
            GITEA_CONFIG_PATH,
            "--work-path",
            GITEA_WORK_PATH,
            "actions",
            "generate-runner-token",
            "--scope",
            REPO_SCOPE,
        ],
        30,
    )
    stdout = str(result.get("stdout") or "")
    stdout_lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    value = stdout_lines[-1] if stdout_lines else ""
    summary = command_summary(result)
    summary["stdout_line_count"] = len(stdout_lines)
    if result.get("exit_code") == 0 and value:
        summary["stdout_redacted"] = "[REDACTED_REGISTRATION_TOKEN]"
    elif stdout:
        summary["stdout_redacted"] = "[REDACTED_NON_TOKEN_OUTPUT]"
    else:
        summary["stdout_redacted"] = ""
    if result.get("exit_code") != 0:
        return None, summary
    return (value if value else None), summary


def execute_recovery(
    *,
    approval_phrase: str | None,
    runner: CommandRunner = run_command,
    sleep_fn: SleepFn = time.sleep,
    wait_seconds: float = 8.0,
) -> dict[str, Any]:
    blockers: list[str] = []
    steps: list[dict[str, Any]] = []
    if approval_phrase != APPROVAL_PHRASE:
        blockers.append("exact_runner_recovery_approval_phrase_required")
        return build_report(mode="execute", blockers=blockers, steps=steps, runner=runner)

    token, token_step = generate_registration_token(runner=runner)
    steps.append({"step": "generate_registration_token", **token_step})
    if not token:
        blockers.append("runner_registration_token_generation_failed")
        return build_report(mode="execute", blockers=blockers, steps=steps, runner=runner)

    if docker_inspect_exists(RUNNER_NAME, runner=runner):
        result = runner(["docker", "rm", "-f", RUNNER_NAME], 30)
        steps.append({"step": "remove_existing_runner_container", **command_summary(result)})
        if result["exit_code"] != 0:
            blockers.append("unable_to_remove_existing_runner_container")
            return build_report(mode="execute", blockers=blockers, steps=steps, runner=runner)

    result = runner(["docker", "volume", "create", RUNNER_VOLUME], 30)
    steps.append({"step": "ensure_runner_volume", **command_summary(result)})
    if result["exit_code"] != 0:
        blockers.append("unable_to_create_runner_volume")
        return build_report(mode="execute", blockers=blockers, steps=steps, runner=runner)

    run_args = [
        "docker",
        "run",
        "-d",
        "--name",
        RUNNER_NAME,
        "--network",
        RUNNER_NETWORK,
        "-v",
        f"{RUNNER_VOLUME}:/data",
        "-v",
        "/var/run/docker.sock:/var/run/docker.sock",
        "-e",
        f"GITEA_INSTANCE_URL={INSTANCE_URL}",
        "-e",
        f"GITEA_RUNNER_REGISTRATION_TOKEN={token}",
        "-e",
        f"GITEA_RUNNER_NAME={RUNNER_NAME}",
        "-e",
        f"GITEA_RUNNER_LABELS={RUNNER_LABELS}",
        RUNNER_IMAGE,
    ]
    result = runner(run_args, 60)
    run_summary = command_summary(result)
    run_summary["env_keys"] = [
        "GITEA_INSTANCE_URL",
        "GITEA_RUNNER_REGISTRATION_TOKEN",
        "GITEA_RUNNER_NAME",
        "GITEA_RUNNER_LABELS",
    ]
    steps.append({"step": "start_runner_container", **run_summary})
    if result["exit_code"] != 0:
        blockers.append("unable_to_start_runner_container")
        return build_report(mode="execute", blockers=blockers, steps=steps, runner=runner)

    sleep_fn(wait_seconds)
    report = build_report(mode="execute", blockers=blockers, steps=steps, runner=runner)
    post = report["runner_inspection"]
    if post["token_empty_loop_detected"]:
        report["blockers"].append("runner_still_reports_token_empty")
    if post["instance_empty_loop_detected"]:
        report["blockers"].append("runner_still_reports_instance_empty")
    report["conclusion"] = "FAIL" if report["blockers"] else "PASS"
    return report


def build_report(
    *,
    mode: str,
    blockers: list[str] | None = None,
    steps: list[dict[str, Any]] | None = None,
    runner: CommandRunner = run_command,
) -> dict[str, Any]:
    blockers = blockers or []
    steps = steps or []
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "mode": mode,
        "approval_phrase_required": APPROVAL_PHRASE,
        "runner_container": RUNNER_NAME,
        "gitea_container": GITEA_CONTAINER,
        "runner_image": RUNNER_IMAGE,
        "runner_network": RUNNER_NETWORK,
        "runner_volume": RUNNER_VOLUME,
        "runner_state_file": RUNNER_STATE_FILE,
        "runner_daemon_command": RUNNER_DAEMON_COMMAND,
        "runner_labels": RUNNER_LABELS,
        "repo_scope": REPO_SCOPE,
        "planned_env_keys": [
            "GITEA_INSTANCE_URL",
            "GITEA_RUNNER_REGISTRATION_TOKEN",
            "GITEA_RUNNER_NAME",
            "GITEA_RUNNER_LABELS",
        ],
        "forbidden_env_keys": ["GITEA_RUNNER_TOKEN"],
        "runner_inspection": inspect_runner(runner=runner),
        "steps": steps,
        "blockers": blockers,
        "secrets_redacted": True,
        "workflow_dispatch_invoked": False,
        "pr_mutation_invoked": False,
        "merge_invoked": False,
        "direct_db_token_insertion_invoked": False,
    }
    report["conclusion"] = "FAIL" if blockers else "PASS"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gated local Gitea act_runner recovery for crypto_bot."
    )
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approval-phrase")
    parser.add_argument("--wait-seconds", type=float, default=8.0)
    parser.add_argument("--format", choices=["json"], default="json")
    args = parser.parse_args()
    if args.inspect and args.execute:
        parser.error("choose either --inspect or --execute")
    if args.execute:
        report = execute_recovery(
            approval_phrase=args.approval_phrase,
            wait_seconds=args.wait_seconds,
        )
    else:
        report = build_report(mode="inspect")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["conclusion"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
