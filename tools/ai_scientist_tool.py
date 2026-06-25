import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from hermes_constants import get_hermes_home
from tools.ai_scientist_env import build_ai_scientist_env, resolve_ai_scientist_run_config
from tools.ai_scientist_deps import ensure_ai_scientist_deps
from tools.registry import registry

logger = logging.getLogger(__name__)

HERMES_ROOT = Path(__file__).parent.parent
AI_SCIENTIST_DIR = HERMES_ROOT / "vendor" / "openclaw-mirror" / "AI-Scientist"
AI_SCIENTIST_ENTRYPOINT = AI_SCIENTIST_DIR / "launch_scientist.py"
AI_SCIENTIST_LAUNCHER = HERMES_ROOT / "scripts" / "ai_scientist_launcher.py"


def check_ai_scientist_available() -> bool:
    """Check if AI-Scientist vendor tree is present."""
    return AI_SCIENTIST_ENTRYPOINT.is_file() and AI_SCIENTIST_LAUNCHER.is_file()


def _copy_results_to_hermes_home(experiment: str, dest: Path) -> None:
    """Upstream writes to vendor/results/<experiment>; mirror into HERMES_HOME."""
    src = AI_SCIENTIST_DIR / "results" / experiment
    if not src.is_dir():
        return
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)


def ai_scientist_research(
    experiment: str = "nanoGPT_lite",
    num_ideas: int = 2,
    model: str = "gpt-4o-mini",
    results_dir: str | None = None,
    task_id: str | None = None,
    use_gpu: bool = True,
    skip_novelty_check: bool = True,
    engine: str = "openalex",
) -> str:
    """
    Launches an AI-Scientist research run via upstream launch_scientist.py.

    Results are written under vendor/.../results/<experiment> and mirrored to
    HERMES_HOME when results_dir is omitted.
    """
    if not results_dir:
        results_dir = str(get_hermes_home() / "evolution" / "ai_scientist" / (task_id or "research_run"))

    results_path = Path(results_dir).absolute()
    results_path.mkdir(parents=True, exist_ok=True)

    try:
        ensure_ai_scientist_deps(prompt=False)
    except Exception as exc:
        return json.dumps(
            {
                "success": False,
                "error": (
                    "AI-Scientist runtime deps missing (aider-chat). "
                    "Run: uv sync --extra ai-scientist"
                ),
                "detail": str(exc),
            }
        )

    run_config = resolve_ai_scientist_run_config(model)
    sakana_model = run_config.get("sakana_model") or model
    if not run_config.get("has_credentials"):
        return json.dumps(
            {
                "success": False,
                "error": (
                    "No Hermes-bridged credentials for AI-Scientist. "
                    "Run hermes auth / hermes setup for Codex OAuth, Nous, Groq, Gemini, or set API keys in ~/.hermes/.env"
                ),
                "credential_status": {
                    "model": model,
                    "sakana_model": sakana_model,
                    "provider_id": run_config.get("provider_id"),
                },
            }
        )

    cmd = [
        sys.executable,
        str(AI_SCIENTIST_LAUNCHER),
        "--model",
        sakana_model,
        "--experiment",
        experiment,
        "--num-ideas",
        str(num_ideas),
        "--engine",
        engine,
    ]
    if skip_novelty_check:
        cmd.append("--skip-novelty-check")

    env = build_ai_scientist_env(model=model)
    if use_gpu:
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    logger.info("Launching AI-Scientist run: %s", " ".join(cmd))

    try:
        res = subprocess.run(
            cmd,
            cwd=str(AI_SCIENTIST_DIR),
            capture_output=True,
            text=True,
            timeout=7200,
            env=env,
        )
        if res.returncode == 0:
            _copy_results_to_hermes_home(experiment, results_path)

        return json.dumps(
            {
                "success": res.returncode == 0,
                "stdout_tail": res.stdout[-2000:],
                "stderr_tail": res.stderr[-2000:],
                "results_dir": str(results_path),
                "vendor_results_dir": str(AI_SCIENTIST_DIR / "results" / experiment),
                "exit_code": res.returncode,
                "sakana_model": sakana_model,
                "hermes_provider": run_config.get("provider_id"),
                "routing": run_config.get("routing"),
            }
        )
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "AI-Scientist run timed out after 2 hours."})
    except Exception as e:
        logger.error("AI-Scientist execution failed: %s", e)
        return json.dumps({"error": str(e)})


registry.register(
    name="ai_scientist_research",
    toolset="self_evolution",
    schema={
        "name": "ai_scientist_research",
        "description": "Execute an AI-Scientist research run to autonomously explore and test new ideas.",
        "parameters": {
            "type": "object",
            "properties": {
                "experiment": {
                    "type": "string",
                    "description": "Experiment template name (e.g. nanoGPT_lite, nanoGPT, nc_kan).",
                    "default": "nanoGPT_lite",
                },
                "num_ideas": {
                    "type": "integer",
                    "description": "Number of ideas to generate and evaluate.",
                    "default": 2,
                },
                "model": {
                    "type": "string",
                    "description": (
                        "Sakana model alias or 'auto' to route via Hermes OAuth/free-tier "
                        "(Codex, Nous, NVIDIA, Groq, xAI, Gemini, Anthropic, Ollama)."
                    ),
                    "default": "auto",
                },
                "results_dir": {
                    "type": "string",
                    "description": "Directory to mirror results into. Defaults to HERMES_HOME/evolution/ai_scientist/.",
                },
                "use_gpu": {
                    "type": "boolean",
                    "description": "Set CUDA_VISIBLE_DEVICES=0 for the research run.",
                    "default": True,
                },
                "skip_novelty_check": {
                    "type": "boolean",
                    "description": "Skip literature novelty check (faster harness runs).",
                    "default": True,
                },
                "engine": {
                    "type": "string",
                    "description": "Literature search backend when novelty check is enabled.",
                    "enum": ["semanticscholar", "openalex"],
                    "default": "openalex",
                },
            },
        },
    },
    handler=lambda args, **kw: ai_scientist_research(
        experiment=args.get("experiment", "nanoGPT_lite"),
        num_ideas=args.get("num_ideas", 2),
        model=args.get("model", "auto"),
        results_dir=args.get("results_dir"),
        task_id=kw.get("task_id"),
        use_gpu=args.get("use_gpu", True),
        skip_novelty_check=args.get("skip_novelty_check", True),
        engine=args.get("engine", "openalex"),
    ),
    check_fn=check_ai_scientist_available,
    emoji="🧪",
)
