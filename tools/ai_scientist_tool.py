import json
import os
import subprocess
import logging
from pathlib import Path
from tools.registry import registry
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Base directory for AI-Scientist (assumed relative to hermes-agent root)
HERMES_ROOT = Path(__file__).parent.parent
AI_SCIENTIST_DIR = HERMES_ROOT / "vendor" / "openclaw-mirror" / "AI-Scientist"
AI_SCIENTIST_ENTRYPOINT = AI_SCIENTIST_DIR / "launch_scientist.py"

def check_ai_scientist_available() -> bool:
    """Check if AI-Scientist submodule is present."""
    return AI_SCIENTIST_ENTRYPOINT.is_file()

def ai_scientist_research(
    experiment: str = "nc_kan", 
    num_ideas: int = 2, 
    model: str = "anthropic/claude-3-5-sonnet-20241022",
    results_dir: str = None, 
    task_id: str = None,
    use_gpu: bool = True
) -> str:
    """
    Launches an AI-Scientist research run.
    
    Args:
        experiment: Template name or experiment identifier.
        num_ideas: Number of ideas to generate and test.
        model: LLM for idea generation and execution.
        results_dir: Directory to save results.
        task_id: Session identifier.
        use_gpu: Whether to prioritize GPU (RTX 3060) usage.
    """
    if not results_dir:
        results_dir = str(get_hermes_home() / "evolution" / "ai_scientist" / (task_id or "research_run"))
    
    results_path = Path(results_dir).absolute()
    results_path.mkdir(parents=True, exist_ok=True)

    # Prepare command for launch_scientist.py
    cmd = [
        "python", "launch_scientist.py", 
        "--model", model,
        "--experiment", experiment,
        "--num-ideas", str(num_ideas),
        "--results-dir", str(results_path)
    ]
    
    # Environment variables for GPU acceleration and API keys
    env = os.environ.copy()
    if use_gpu:
        env["CUDA_VISIBLE_DEVICES"] = "0" # RTX 3060
    
    logger.info(f"Launching AI-Scientist run: {' '.join(cmd)}")
    
    try:
        # Research runs are extremely long. 
        # In a real growth pulse, we might want to background this using process_registry.
        res = subprocess.run(
            cmd, 
            cwd=str(AI_SCIENTIST_DIR), 
            capture_output=True, 
            text=True, 
            timeout=7200, # 2 hours
            env=env
        )
        
        success = res.returncode == 0
        return json.dumps({
            "success": success,
            "stdout_tail": res.stdout[-2000:],
            "stderr_tail": res.stderr[-2000:],
            "results_dir": str(results_path),
            "exit_code": res.returncode
        })
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "AI-Scientist run timed out after 2 hours."})
    except Exception as e:
        logger.error(f"AI-Scientist execution failed: {e}")
        return json.dumps({"error": str(e)})

# Register the tool
registry.register(
    name="ai_scientist_research",
    toolset="self_evolution",
    schema={
        "name": "ai_scientist_research",
        "description": "Execute an AI-Scientist research run to autonomously explore and test new ideas or formal proofs.",
        "parameters": {
            "type": "object",
            "properties": {
                "experiment": {
                    "type": "string", 
                    "description": "The experiment template name (e.g. 'nc_kan' or 'hermes_self_evolve').",
                    "default": "nc_kan"
                },
                "num_ideas": {
                    "type": "integer", 
                    "description": "Number of ideas to generate and evaluate.",
                    "default": 2
                },
                "model": {
                    "type": "string", 
                    "description": "Model identifier for the researcher (e.g. 'anthropic/claude-3-5-sonnet-20241022').",
                    "default": "anthropic/claude-3-5-sonnet-20241022"
                },
                "results_dir": {
                    "type": "string", 
                    "description": "Absolute path to results directory. Defaults to HERMES_HOME/evolution/ai_scientist/."
                },
                "use_gpu": {
                    "type": "boolean",
                    "description": "Enable GPU (RTX 3060) usage for the research run.",
                    "default": True
                }
            }
        }
    },
    handler=lambda args, **kw: ai_scientist_research(
        experiment=args.get("experiment", "nc_kan"),
        num_ideas=args.get("num_ideas", 2),
        model=args.get("model", "anthropic/claude-3-5-sonnet-20241022"),
        results_dir=args.get("results_dir"),
        task_id=kw.get("task_id"),
        use_gpu=args.get("use_gpu", True)
    ),
    check_fn=check_ai_scientist_available,
    emoji="🧪"
)
