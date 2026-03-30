"""ANIMA Intelligence Compiler tools for ROS2 robotics pipeline deployment.

ANIMA is an offline robotics intelligence compiler that transforms natural
language task descriptions into validated, deployable ROS2 pipelines. It
provides:

- 96-module registry (perception, planning, control, foundation models)
- 3-gate validation (schema, semantic, resolution)
- CSP constraint solver (VRAM budget, platform compat, license audit)
- Code generation (docker-compose, ROS2 launch, BehaviorTree XML)
- Safety supervisor with continuous health monitoring and e-stop

Registers six LLM-callable tools:
- ``anima_compile``    -- compile a natural language task into a ROS2 pipeline
- ``anima_validate``   -- validate a task spec against the 3-gate system
- ``anima_deploy``     -- deploy a compiled pipeline to Docker containers
- ``anima_status``     -- check health of running ANIMA modules
- ``anima_stop``       -- stop running pipeline (graceful or e-stop)
- ``anima_registry``   -- search the module registry by capability

The ANIMA compiler server must be running (default: http://localhost:3000).
Set ``ANIMA_COMPILER_URL`` to override. Optionally set ``ANIMA_COMPILER_TOKEN``
for authenticated access.

Docs: https://github.com/RobotFlow-Labs/anima-infra
"""

import json
import logging
import os
import shutil
import subprocess
from typing import Any, Dict, Optional

from tools.registry import registry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_COMPILER_URL = "http://localhost:3000"


def _get_config():
    """Return (compiler_url, token) from env vars at call time."""
    return (
        os.getenv("ANIMA_COMPILER_URL", _DEFAULT_COMPILER_URL).rstrip("/"),
        os.getenv("ANIMA_COMPILER_TOKEN", ""),
    )


def _has_anima() -> bool:
    """Check if ANIMA compiler API or CLI is reachable."""
    # Check env var first (API mode)
    if os.getenv("ANIMA_COMPILER_URL"):
        return True
    # Check if CLI is installed
    if shutil.which("anima"):
        return True
    # Check if default server is reachable (best effort)
    try:
        import httpx
        resp = httpx.get(f"{_DEFAULT_COMPILER_URL}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _headers(token: str = "") -> Dict[str, str]:
    """Build request headers with optional auth."""
    h = {"Content-Type": "application/json"}
    if not token:
        _, token = _get_config()
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _api_get(path: str, timeout: float = 10) -> Dict[str, Any]:
    """GET request to ANIMA compiler API."""
    import httpx
    url, token = _get_config()
    resp = httpx.get(f"{url}{path}", headers=_headers(token), timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _api_post(path: str, body: Dict, timeout: float = 30) -> Dict[str, Any]:
    """POST request to ANIMA compiler API."""
    import httpx
    url, token = _get_config()
    resp = httpx.post(
        f"{url}{path}",
        headers=_headers(token),
        json=body,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# CLI fallback helpers
# ---------------------------------------------------------------------------

def _cli_available() -> bool:
    return shutil.which("anima") is not None


def _run_cli(args: list, timeout: int = 60) -> str:
    """Run an ANIMA CLI command and return stdout."""
    result = subprocess.run(
        ["anima"] + args,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"anima {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Tool: anima_compile
# ---------------------------------------------------------------------------

def _handle_compile(args: Dict[str, Any], **kwargs) -> str:
    """Compile a natural language task into a deployable ROS2 pipeline."""
    try:
        task = args.get("task", "")
        platform = args.get("platform", "x86_64")
        vram_budget_mb = args.get("vram_budget_mb", 23000)
        dry_run = args.get("dry_run", False)

        if not task:
            return json.dumps({"error": "task is required"})

        if len(task) > 1000:
            return json.dumps({"error": "Task description exceeds 1000 character limit"})

        # Try API first, fall back to CLI
        try:
            result = _api_post("/compile", {
                "task": task,
                "platform": platform,
                "hardware_budget_mb": vram_budget_mb,
                "dry_run": dry_run,
            }, timeout=30)
        except Exception as api_err:
            if _cli_available():
                logger.info("API unavailable, falling back to CLI: %s", api_err)
                cli_args = ["compose", "--task", task, "--platform", platform]
                if dry_run:
                    cli_args.append("--dry-run")
                output = _run_cli(cli_args)
                result = {"status": "compiled", "output": output}
            else:
                raise

        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        logger.error("anima_compile failed: %s", e)
        return json.dumps({"error": str(e)})


COMPILE_SCHEMA = {
    "name": "anima_compile",
    "description": (
        "Compile a natural language robotics task into a deployable ROS2 pipeline "
        "using the ANIMA Intelligence Compiler. The compiler selects optimal modules "
        "from a 96-module registry, validates the pipeline through 3 gates (schema, "
        "semantic, resolution), solves constraints (VRAM, platform, licenses), and "
        "generates docker-compose, ROS2 launch files, and BehaviorTree XML. "
        "Example tasks: 'detect objects on the table and pick the red ones', "
        "'navigate to the kitchen while avoiding obstacles', "
        "'estimate depth and segment the scene for manipulation planning'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": (
                    "Natural language description of the robotics task to compile. "
                    "Be specific about what perception, planning, or control is needed."
                ),
            },
            "platform": {
                "type": "string",
                "enum": ["x86_64", "arm64", "jetson"],
                "default": "x86_64",
                "description": "Target hardware platform for the pipeline.",
            },
            "vram_budget_mb": {
                "type": "integer",
                "default": 23000,
                "description": "Total GPU VRAM budget in MB. Default 23000 (NVIDIA L4).",
            },
            "dry_run": {
                "type": "boolean",
                "default": False,
                "description": "If true, validate and solve but don't generate artifacts.",
            },
        },
        "required": ["task"],
    },
}


# ---------------------------------------------------------------------------
# Tool: anima_validate
# ---------------------------------------------------------------------------

def _handle_validate(args: Dict[str, Any], **kwargs) -> str:
    """Validate a task spec through ANIMA's 3-gate validation system."""
    try:
        task_spec = args.get("task_spec", "")
        if not task_spec:
            return json.dumps({"error": "task_spec is required (YAML or JSON string)"})

        result = _api_post("/validate", {"task_spec": task_spec}, timeout=15)
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        logger.error("anima_validate failed: %s", e)
        return json.dumps({"error": str(e)})


VALIDATE_SCHEMA = {
    "name": "anima_validate",
    "description": (
        "Validate a robotics task specification through ANIMA's 3-gate system: "
        "Gate 1 checks schema correctness, Gate 2 checks semantic validity "
        "(capabilities exist, VRAM fits, platform compatible), Gate 3 checks "
        "resolution (no cycles, types match, all ports connected). "
        "Use this to check if a pipeline spec is valid before compiling."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_spec": {
                "type": "string",
                "description": "Task specification as YAML or JSON string to validate.",
            },
        },
        "required": ["task_spec"],
    },
}


# ---------------------------------------------------------------------------
# Tool: anima_deploy
# ---------------------------------------------------------------------------

def _handle_deploy(args: Dict[str, Any], **kwargs) -> str:
    """Deploy a compiled pipeline to Docker containers."""
    try:
        compose_file = args.get("compose_file", "")
        gpu_ids = args.get("gpu_ids", "")

        if not compose_file:
            return json.dumps({"error": "compose_file path is required"})

        if not os.path.isfile(compose_file):
            return json.dumps({"error": f"Compose file not found: {compose_file}"})

        # Use CLI if available, otherwise docker compose directly
        if _cli_available():
            cli_args = ["deploy", "--compose", compose_file]
            if gpu_ids:
                cli_args.extend(["--gpus", gpu_ids])
            output = _run_cli(cli_args, timeout=120)
            result = {"status": "deployed", "output": output}
        else:
            # Direct docker compose
            env = os.environ.copy()
            if gpu_ids:
                env["CUDA_VISIBLE_DEVICES"] = gpu_ids
            cmd = ["docker", "compose", "-f", compose_file, "up", "-d"]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
            if proc.returncode != 0:
                return json.dumps({"error": f"Deploy failed: {proc.stderr.strip()}"})
            result = {"status": "deployed", "output": proc.stdout.strip()}

        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        logger.error("anima_deploy failed: %s", e)
        return json.dumps({"error": str(e)})


DEPLOY_SCHEMA = {
    "name": "anima_deploy",
    "description": (
        "Deploy a compiled ANIMA pipeline to Docker containers with GPU pinning. "
        "Takes a docker-compose file generated by anima_compile and starts all "
        "module containers. Each container runs a ROS2 node with real model inference, "
        "publishes health at 1Hz, and exposes /health, /ready, /predict HTTP endpoints."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "compose_file": {
                "type": "string",
                "description": "Path to the docker-compose YAML file generated by anima_compile.",
            },
            "gpu_ids": {
                "type": "string",
                "default": "",
                "description": "Comma-separated GPU IDs to use (e.g. '0,1,2'). Empty = all available.",
            },
        },
        "required": ["compose_file"],
    },
}


# ---------------------------------------------------------------------------
# Tool: anima_status
# ---------------------------------------------------------------------------

def _handle_status(args: Dict[str, Any], **kwargs) -> str:
    """Check health status of running ANIMA modules."""
    try:
        module_name = args.get("module_name", "")

        # Try API first
        try:
            health = _api_get("/health", timeout=5)
            modules_loaded = health.get("modules_loaded", 0)
        except Exception:
            health = {"status": "compiler_unreachable"}
            modules_loaded = 0

        # Check individual module containers
        modules = {}
        if module_name:
            # Check specific module
            names = [module_name]
        else:
            # Check all running ANIMA containers
            try:
                proc = subprocess.run(
                    ["docker", "ps", "--filter", "name=anima-", "--format", "{{.Names}}"],
                    capture_output=True, text=True, timeout=10,
                )
                names = [n.replace("anima-", "") for n in proc.stdout.strip().split("\n") if n]
            except Exception:
                names = []

        for name in names:
            try:
                import httpx
                # Module ports follow convention: bestla=8081, centaur=8082, etc.
                # Try common ports or use discovery
                for port in range(8081, 8099):
                    try:
                        resp = httpx.get(
                            f"http://localhost:{port}/health",
                            timeout=2,
                        )
                        data = resp.json()
                        if data.get("module", "").lower() == name.lower() or \
                           name.lower() in data.get("module", "").lower():
                            modules[name] = {
                                "status": data.get("status", "unknown"),
                                "port": port,
                                "gpu_vram_mb": data.get("gpu_vram_mb", 0),
                                "uptime_s": data.get("uptime_s", 0),
                            }
                            break
                    except Exception:
                        continue
                if name not in modules:
                    modules[name] = {"status": "unreachable"}
            except Exception:
                modules[name] = {"status": "error"}

        return json.dumps({
            "compiler": health,
            "modules_loaded": modules_loaded,
            "running_modules": modules,
        }, ensure_ascii=False)

    except Exception as e:
        logger.error("anima_status failed: %s", e)
        return json.dumps({"error": str(e)})


STATUS_SCHEMA = {
    "name": "anima_status",
    "description": (
        "Check the health status of the ANIMA compiler and running module containers. "
        "Returns compiler status, number of loaded modules, and per-container health "
        "(status, GPU VRAM usage, uptime). Use without arguments to check all running "
        "modules, or specify a module name for a specific check."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "module_name": {
                "type": "string",
                "default": "",
                "description": "Specific module to check (e.g. 'centaur', 'morpheus'). Empty = all.",
            },
        },
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# Tool: anima_stop
# ---------------------------------------------------------------------------

def _handle_stop(args: Dict[str, Any], **kwargs) -> str:
    """Stop running ANIMA pipeline — graceful shutdown or emergency stop."""
    try:
        emergency = args.get("emergency", False)
        compose_file = args.get("compose_file", "")

        if emergency:
            # E-stop: call safety supervisor directly
            try:
                result = _api_post("/estop", {"reason": "user_requested"}, timeout=5)
                return json.dumps({"status": "emergency_stopped", "details": result})
            except Exception:
                # Fallback: kill all ANIMA containers directly
                proc = subprocess.run(
                    ["docker", "ps", "-q", "--filter", "name=anima-"],
                    capture_output=True, text=True, timeout=10,
                )
                container_ids = proc.stdout.strip().split("\n")
                container_ids = [c for c in container_ids if c]
                if container_ids:
                    subprocess.run(
                        ["docker", "kill"] + container_ids,
                        capture_output=True, timeout=15,
                    )
                return json.dumps({
                    "status": "emergency_stopped",
                    "containers_killed": len(container_ids),
                })

        # Graceful shutdown
        if compose_file and os.path.isfile(compose_file):
            proc = subprocess.run(
                ["docker", "compose", "-f", compose_file, "down"],
                capture_output=True, text=True, timeout=60,
            )
            return json.dumps({
                "status": "stopped",
                "output": proc.stdout.strip(),
            })

        # No compose file — stop all ANIMA containers
        if _cli_available():
            output = _run_cli(["stop"], timeout=30)
            return json.dumps({"status": "stopped", "output": output})

        # Direct docker stop
        proc = subprocess.run(
            ["docker", "ps", "-q", "--filter", "name=anima-"],
            capture_output=True, text=True, timeout=10,
        )
        container_ids = [c for c in proc.stdout.strip().split("\n") if c]
        if container_ids:
            subprocess.run(
                ["docker", "stop"] + container_ids,
                capture_output=True, timeout=60,
            )
        return json.dumps({
            "status": "stopped",
            "containers_stopped": len(container_ids),
        })

    except Exception as e:
        logger.error("anima_stop failed: %s", e)
        return json.dumps({"error": str(e)})


STOP_SCHEMA = {
    "name": "anima_stop",
    "description": (
        "Stop a running ANIMA pipeline. Use emergency=true for immediate e-stop "
        "(kills all containers instantly, no graceful shutdown). Use emergency=false "
        "(default) for graceful shutdown that lets modules finish current inference. "
        "IMPORTANT: Use emergency stop if the robot is behaving unsafely."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "emergency": {
                "type": "boolean",
                "default": False,
                "description": "If true, kill all containers immediately (e-stop). Use for safety.",
            },
            "compose_file": {
                "type": "string",
                "default": "",
                "description": "Path to compose file for graceful shutdown. Empty = stop all ANIMA containers.",
            },
        },
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# Tool: anima_registry
# ---------------------------------------------------------------------------

def _handle_registry(args: Dict[str, Any], **kwargs) -> str:
    """Search the ANIMA module registry by capability or name."""
    try:
        query = args.get("query", "")
        capability = args.get("capability", "")

        if not query and not capability:
            # List all modules
            try:
                result = _api_get("/registry/modules", timeout=10)
                return json.dumps(result, ensure_ascii=False)
            except Exception:
                if _cli_available():
                    output = _run_cli(["registry", "list"])
                    return json.dumps({"modules": output})
                raise

        # Search by capability or name
        try:
            params = {}
            if capability:
                params["capability"] = capability
            if query:
                params["q"] = query

            import httpx
            url, token = _get_config()
            resp = httpx.get(
                f"{url}/registry/modules",
                headers=_headers(token),
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            result = resp.json()
        except Exception:
            if _cli_available():
                cli_args = ["registry", "search"]
                if capability:
                    cli_args.extend(["--capability", capability])
                if query:
                    cli_args.append(query)
                output = _run_cli(cli_args)
                result = {"results": output}
            else:
                raise

        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        logger.error("anima_registry failed: %s", e)
        return json.dumps({"error": str(e)})


REGISTRY_SCHEMA = {
    "name": "anima_registry",
    "description": (
        "Search the ANIMA module registry (96 robotics AI modules). "
        "Modules cover perception (detection, depth, segmentation, SLAM, place recognition), "
        "planning (manipulation, navigation, task planning), control (diffusion policy, VLA), "
        "and foundation models (world models, scene flow). "
        "Search by capability type (e.g. 'perception.vision') or by name (e.g. 'morpheus'). "
        "Returns module metadata including capabilities, hardware requirements, and performance."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "default": "",
                "description": "Free-text search query (module name, description keywords).",
            },
            "capability": {
                "type": "string",
                "default": "",
                "description": (
                    "Filter by capability type: perception.vision, perception.depth, "
                    "perception.segmentation, perception.slam, planning.manipulation, "
                    "planning.navigation, control.diffusion_policy, foundation.world_model, etc."
                ),
            },
        },
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="anima_compile",
    toolset="anima",
    schema=COMPILE_SCHEMA,
    handler=lambda args, **kw: _handle_compile(args, **kw),
    check_fn=_has_anima,
    requires_env=[],
    emoji="🤖",
)

registry.register(
    name="anima_validate",
    toolset="anima",
    schema=VALIDATE_SCHEMA,
    handler=lambda args, **kw: _handle_validate(args, **kw),
    check_fn=_has_anima,
    requires_env=[],
    emoji="✅",
)

registry.register(
    name="anima_deploy",
    toolset="anima",
    schema=DEPLOY_SCHEMA,
    handler=lambda args, **kw: _handle_deploy(args, **kw),
    check_fn=_has_anima,
    requires_env=[],
    emoji="🚀",
)

registry.register(
    name="anima_status",
    toolset="anima",
    schema=STATUS_SCHEMA,
    handler=lambda args, **kw: _handle_status(args, **kw),
    check_fn=_has_anima,
    requires_env=[],
    emoji="📊",
)

registry.register(
    name="anima_stop",
    toolset="anima",
    schema=STOP_SCHEMA,
    handler=lambda args, **kw: _handle_stop(args, **kw),
    check_fn=_has_anima,
    requires_env=[],
    emoji="🛑",
)

registry.register(
    name="anima_registry",
    toolset="anima",
    schema=REGISTRY_SCHEMA,
    handler=lambda args, **kw: _handle_registry(args, **kw),
    check_fn=_has_anima,
    requires_env=[],
    emoji="📦",
)
