"""ShinkaEvolve batch runner — Hermes tool surface for vendor/shinka-osint."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from hermes_constants import get_hermes_home
from tools.registry import registry
from tools.shinka_evolve_env import build_shinka_env, resolve_shinka_run_config

logger = logging.getLogger(__name__)

HERMES_ROOT = Path(__file__).resolve().parent.parent
# Canonical checkout on this fork: vendor/shinka-osint (not openclaw-mirror/ShinkaEvolve).
SHINKA_DIR = HERMES_ROOT / "vendor" / "shinka-osint"
SHINKA_PACKAGE_INIT = SHINKA_DIR / "shinka" / "__init__.py"
SHINKA_CLI_RUN = SHINKA_DIR / "shinka" / "cli" / "run.py"


def check_shinka_available() -> bool:
    """Check if ShinkaEvolve package is present under vendor/shinka-osint."""
    return SHINKA_PACKAGE_INIT.is_file() and SHINKA_CLI_RUN.is_file()


def shinka_run_batch(
    task_dir: str,
    num_generations: int = 10,
    results_dir: str | None = None,
    task_id: str | None = None,
    use_gpu: bool = True,
    model: str = "auto",
) -> str:
    """
    Run a ShinkaEvolve batch via ``python -m shinka.cli.run``.

    Credentials are bridged from Hermes (``~/.hermes/.env`` + OAuth stores)
    through ``tools.shinka_evolve_env`` — no parallel secret store.
    """
    if not task_dir or not str(task_dir).strip():
        return json.dumps({"success": False, "error": "task_dir is required"})

    if not results_dir:
        results_dir = str(get_hermes_home() / "evolution" / "shinka" / (task_id or "growth_pulse"))

    results_path = Path(results_dir).absolute()
    results_path.mkdir(parents=True, exist_ok=True)

    run_config = resolve_shinka_run_config(model)
    if not run_config.get("has_credentials"):
        return json.dumps(
            {
                "success": False,
                "error": (
                    "No Hermes-bridged credentials for ShinkaEvolve. "
                    "Run hermes auth / hermes setup, or set API keys in ~/.hermes/.env"
                ),
                "credential_status": {
                    "model": model,
                    "provider_id": run_config.get("provider_id"),
                },
            }
        )

    cmd = [
        sys.executable,
        "-m",
        "shinka.cli.run",
        "--task-dir",
        str(task_dir),
        "--results_dir",
        str(results_path),
        "--num_generations",
        str(int(num_generations)),
    ]
    llm_models = run_config.get("llm_models") or []
    if llm_models:
        cmd.extend(["--set", f"evo.llm_models={json.dumps(llm_models)}"])

    env = build_shinka_env(model=model)
    existing_pythonpath = env.get("PYTHONPATH", "")
    path_parts = [str(SHINKA_DIR)]
    if existing_pythonpath:
        path_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(path_parts)

    if use_gpu:
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    logger.info("Launching ShinkaEvolve batch: %s", " ".join(cmd))

    try:
        res = subprocess.run(
            cmd,
            cwd=str(SHINKA_DIR),
            capture_output=True,
            text=True,
            timeout=3600,
            env=env,
        )
        return json.dumps(
            {
                "success": res.returncode == 0,
                "stdout_tail": (res.stdout or "")[-2000:],
                "stderr_tail": (res.stderr or "")[-2000:],
                "results_dir": str(results_path),
                "exit_code": res.returncode,
                "llm_models": llm_models,
                "hermes_provider": run_config.get("provider_id"),
                "routing": run_config.get("routing"),
            }
        )
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "ShinkaEvolve batch timed out after 1 hour."})
    except Exception as e:
        logger.error("ShinkaEvolve execution failed: %s", e)
        return json.dumps({"error": str(e)})


registry.register(
    name="shinka_run",
    toolset="self_evolution",
    schema={
        "name": "shinka_run",
        "description": (
            "Execute a ShinkaEvolve batch to autonomously optimize a component "
            "or proof. Uses Hermes-configured LLM credentials."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task_dir": {
                    "type": "string",
                    "description": (
                        "Path to task directory relative to vendor/shinka-osint "
                        "(e.g. 'examples/circle_packing') or an absolute path "
                        "with evaluate.py + initial.py."
                    ),
                },
                "num_generations": {
                    "type": "integer",
                    "description": "Number of generations to evolve.",
                    "default": 10,
                },
                "results_dir": {
                    "type": "string",
                    "description": (
                        "Absolute path for results. Defaults to "
                        "HERMES_HOME/evolution/shinka/."
                    ),
                },
                "model": {
                    "type": "string",
                    "description": (
                        "LLM alias or 'auto' to route via Hermes OAuth/API keys "
                        "(Codex, Nous, NVIDIA, Groq, Gemini, Anthropic, …)."
                    ),
                    "default": "auto",
                },
                "use_gpu": {
                    "type": "boolean",
                    "description": "Set CUDA_VISIBLE_DEVICES=0 for the evolution batch.",
                    "default": True,
                },
            },
            "required": ["task_dir"],
        },
    },
    handler=lambda args, **kw: shinka_run_batch(
        task_dir=args.get("task_dir"),
        num_generations=args.get("num_generations", 10),
        results_dir=args.get("results_dir"),
        task_id=kw.get("task_id"),
        use_gpu=args.get("use_gpu", True),
        model=args.get("model", "auto"),
    ),
    check_fn=check_shinka_available,
    emoji="🧬",
)
