#!/usr/bin/env python3
"""Kubernetes pod exec tool — run commands inside a pod.

Registers one LLM-callable tool:

- ``pod_exec`` — run a command inside a Kubernetes pod and capture
  its stdout, stderr, and exit code.

Availability (``check_fn``): requires ``kubectl`` on ``$PATH``.
The tool is registered under the ``k8s`` toolset (opt-in via
``hermes tools enable k8s``), so it never appears in the agent's
schema unless the user has enabled Kubernetes tooling.
"""

import json
import logging
import os
import shlex
import shutil
import subprocess
from typing import Optional

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_OUTPUT_CHARS = 50_000  # soft cap on stdout + stderr combined
_DEFAULT_TIMEOUT = 60       # seconds


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def _check_kubectl() -> bool:
    """Tool is only available when ``kubectl`` is on ``$PATH``."""
    return shutil.which("kubectl") is not None


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _build_kubectl_exec_cmd(
    pod: str,
    command: str,
    container: Optional[str] = None,
    namespace: Optional[str] = None,
) -> list[str]:
    """Build the ``kubectl exec`` argument list.

    Everything after ``--`` is the command string split with
    ``shlex.split()`` so the model can pass a single natural-language
    command string (e.g. ``cat /etc/os-release``) and it arrives as
    separate args inside the container.
    """
    cmd = ["kubectl", "exec"]
    if namespace:
        cmd.extend(["-n", namespace])
    if container:
        cmd.extend(["-c", container])
    cmd.append(pod)
    cmd.append("--")
    cmd.extend(shlex.split(command))
    return cmd


def pod_exec_tool(
    pod: str,
    command: str,
    container: Optional[str] = None,
    namespace: Optional[str] = None,
    timeout: int = _DEFAULT_TIMEOUT,
) -> str:
    """Run *command* inside *pod* and return captured output.

    Parameters
    ----------
    pod
        Name of the target pod.
    command
        Shell command to run inside the container (e.g. ``ls -la /app``).
    container
        Container name within the pod (required for multi-container pods).
    namespace
        Kubernetes namespace.  Defaults to the current-context namespace
        (usually ``default``) when omitted.
    timeout
        Max seconds to wait for the command to complete (default: 60).
    """
    if not pod or not command:
        return tool_error("Both 'pod' and 'command' are required.")

    if not _check_kubectl():
        return tool_error(
            "kubectl is not available on $PATH. "
            "Install the Kubernetes CLI first: https://kubernetes.io/docs/tasks/tools/"
        )

    cmd = _build_kubectl_exec_cmd(pod, command, container, namespace)
    logger.debug("Running: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        stdout = result.stdout
        stderr = result.stderr

        # Truncate combined output if it exceeds the cap so we don't blow
        # past the agent's token budget.
        total = len(stdout) + len(stderr)
        if total > _MAX_OUTPUT_CHARS:
            excess = total - _MAX_OUTPUT_CHARS
            # Prefer truncating stderr (usually less informative), but if
            # stdout alone is already over the limit, truncate that too.
            if len(stdout) > _MAX_OUTPUT_CHARS:
                stdout = stdout[:_MAX_OUTPUT_CHARS] + (
                    f"\n[... truncated {excess} chars ...]"
                )
                stderr = ""
            else:
                # Truncate stderr only
                keep = _MAX_OUTPUT_CHARS - len(stdout)
                stderr = stderr[:keep] + (
                    f"\n[... truncated {excess} chars ...]"
                )

        return tool_result(
            success=result.returncode == 0,
            exit_code=result.returncode,
            stdout=stdout,
            stderr=stderr,
        )

    except subprocess.TimeoutExpired:
        return tool_error(
            f"Command timed out after {timeout}s. "
            f"Try a longer timeout or a simpler command.",
        )
    except FileNotFoundError:
        return tool_error(
            "kubectl not found. It was present at startup but disappeared "
            "during execution — check your $PATH or installation.",
        )
    except PermissionError:
        return tool_error(
            "kubectl exists but is not executable. "
            "Fix with: chmod +x $(which kubectl)",
        )
    except OSError as e:
        return tool_error(f"OS error running kubectl: {e}")
    except Exception as e:
        logger.exception("Unexpected error in pod_exec")
        return tool_error(f"Unexpected error: {e}")


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

POD_EXEC_SCHEMA = {
    "name": "pod_exec",
    "description": (
        "Run a command inside a Kubernetes pod and capture its output. "
        "Useful for debugging — inspecting files, checking environment "
        "variables, testing connectivity, or running diagnostic tools "
        "inside a running container. "
        "Returns stdout, stderr, and the exit code."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pod": {
                "type": "string",
                "description": (
                    "Name of the target pod (e.g. 'my-app-7d8f9c6b4c-x2k5j'). "
                    "Use ``kubectl get pods -n <namespace>`` via the terminal "
                    "tool to discover pod names."
                ),
            },
            "command": {
                "type": "string",
                "description": (
                    "Shell command to run inside the container. "
                    "Examples:\n"
                    "- ``ls -la /app``\n"
                    "- ``cat /etc/config/settings.yaml``\n"
                    "- ``env | sort``\n"
                    "- ``curl -s http://localhost:8080/health``\n"
                    "- ``ps aux``\n"
                    "- ``df -h``"
                ),
            },
            "container": {
                "type": "string",
                "description": (
                    "Container name within the pod (required when the pod "
                    "has more than one container). "
                    "Use ``kubectl describe pod <pod> -n <ns>`` to list "
                    "containers if unsure."
                ),
            },
            "namespace": {
                "type": "string",
                "description": (
                    "Kubernetes namespace. Omit to use the current-context "
                    "namespace (typically 'default')."
                ),
            },
            "timeout": {
                "type": "integer",
                "description": (
                    "Max seconds to wait for the command to finish. "
                    "Increase for long-running commands. Default: 60."
                ),
                "default": _DEFAULT_TIMEOUT,
            },
        },
        "required": ["pod", "command"],
    },
}


# ---------------------------------------------------------------------------
# Handler (sync, wraps the tool function)
# ---------------------------------------------------------------------------

def _handle_pod_exec(args: dict, **kw) -> str:
    return pod_exec_tool(
        pod=args.get("pod", ""),
        command=args.get("command", ""),
        container=args.get("container"),
        namespace=args.get("namespace"),
        timeout=args.get("timeout", _DEFAULT_TIMEOUT),
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="pod_exec",
    toolset="k8s",
    schema=POD_EXEC_SCHEMA,
    handler=_handle_pod_exec,
    check_fn=_check_kubectl,
    emoji="☸️",
)
