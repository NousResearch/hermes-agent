import json
import os
import subprocess
import logging
from pathlib import Path
from tools.registry import registry
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Base directory for ShinkaEvolve (assumed relative to hermes-agent root)
HERMES_ROOT = Path(__file__).parent.parent
SHINKA_DIR = HERMES_ROOT / "vendor" / "openclaw-mirror" / "ShinkaEvolve"
SHINKA_PACKAGE_INIT = SHINKA_DIR / "shinka" / "__init__.py"

def check_shinka_available() -> bool:
    """Check if ShinkaEvolve submodule is present."""
    return SHINKA_PACKAGE_INIT.is_file()

def shinka_run_batch(
    task_dir: str, 
    num_generations: int = 10, 
    results_dir: str = None, 
    task_id: str = None,
    use_gpu: bool = True
) -> str:
    """
    Runs a ShinkaEvolve batch to optimize code.
    
    Args:
        task_dir: Relative path to task directory (e.g., 'examples/my_task')
        num_generations: Number of evolution generations.
        results_dir: Directory to save results.
        task_id: Session/Task identifier.
        use_gpu: Whether to prioritize GPU (RTX 3060) usage.
    """
    if not results_dir:
        results_dir = str(get_hermes_home() / "evolution" / "shinka" / (task_id or "growth_pulse"))
    
    # Ensure results_dir is absolute
    results_path = Path(results_dir).absolute()
    results_path.mkdir(parents=True, exist_ok=True)

    # Prepare command
    cmd = [
        "python", "-m", "shinka.cli.run", 
        "--task-dir", task_dir,
        "--results_dir", str(results_path),
        "--num_generations", str(num_generations)
    ]
    
    # Environment variables for GPU acceleration and API keys
    env = os.environ.copy()
    if use_gpu:
        env["CUDA_VISIBLE_DEVICES"] = "0" # RTX 3060
    
    logger.info(f"Launching ShinkaEvolve batch: {' '.join(cmd)}")
    
    try:
        # Run as a foreground process with a long timeout (1 hour for evolution batch)
        # In a real growth pulse, we might want to background this.
        res = subprocess.run(
            cmd, 
            cwd=str(SHINKA_DIR), 
            capture_output=True, 
            text=True, 
            timeout=3600,
            env=env
        )
        
        success = res.returncode == 0
        return json.dumps({
            "success": success,
            "stdout_tail": res.stdout[-1000:],
            "stderr_tail": res.stderr[-1000:],
            "results_dir": str(results_path),
            "exit_code": res.returncode
        })
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "ShinkaEvolve batch timed out after 1 hour."})
    except Exception as e:
        logger.error(f"ShinkaEvolve execution failed: {e}")
        return json.dumps({"error": str(e)})

# Register the tool
registry.register(
    name="shinka_run",
    toolset="self_evolution",
    schema={
        "name": "shinka_run",
        "description": "Execute a ShinkaEvolve batch to autonomously optimize a component or proof.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_dir": {
                    "type": "string", 
                    "description": "Path to task directory (relative to ShinkaEvolve root, e.g. 'examples/nc_kan_proof')."
                },
                "num_generations": {
                    "type": "integer", 
                    "description": "Number of generations to evolve.",
                    "default": 10
                },
                "results_dir": {
                    "type": "string", 
                    "description": "Absolute path to results directory. Defaults to HERMES_HOME/evolution/shinka/."
                },
                "use_gpu": {
                    "type": "boolean",
                    "description": "Enable GPU (RTX 3060) usage for the evolution batch.",
                    "default": True
                }
            },
            "required": ["task_dir"]
        }
    },
    handler=lambda args, **kw: shinka_run_batch(
        task_dir=args.get("task_dir"),
        num_generations=args.get("num_generations", 10),
        results_dir=args.get("results_dir"),
        task_id=kw.get("task_id"),
        use_gpu=args.get("use_gpu", True)
    ),
    check_fn=check_shinka_available,
    emoji="🧬"
)
