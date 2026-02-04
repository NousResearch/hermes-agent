#!/usr/bin/env python3
"""
RL Training Tools Module

This module provides tools for running RL training through Tinker-Atropos.
Communicates with the RL API server (rl_api_server.py) to manage:
- Environment discovery and selection
- Configuration management
- Training run lifecycle
- WandB metrics monitoring
- Inference-only testing

Required environment variables:
- TINKER_API_KEY: API key for Tinker service
- WANDB_API_KEY: API key for Weights & Biases metrics

Optional environment variables:
- RL_API_URL: URL of the RL API server (default: http://localhost:8080)
- WANDB_ENTITY: WandB entity/team name
- WANDB_PROJECT: Default WandB project name

Usage:
    from tools.rl_training_tool import (
        rl_list_environments,
        rl_select_environment,
        rl_get_current_config,
        rl_edit_config,
        rl_start_training,
        rl_check_status,
        rl_stop_training,
        rl_get_results,
        rl_test_inference,
    )
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

import aiohttp

# ============================================================================
# Configuration
# ============================================================================

# Default RL API server URL (can be overridden via environment variable)
RL_API_URL = os.getenv("RL_API_URL", "http://localhost:8080")

# Rate limiting for status checks (30 minutes in seconds)
MIN_STATUS_CHECK_INTERVAL = 30 * 60
_last_status_check: Dict[str, float] = {}


# ============================================================================
# Helper Functions
# ============================================================================

async def _make_request(
    method: str,
    endpoint: str,
    data: Optional[Dict] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Make an HTTP request to the RL API server."""
    url = f"{RL_API_URL}{endpoint}"
    
    async with aiohttp.ClientSession() as session:
        try:
            if method == "GET":
                async with session.get(url, timeout=timeout) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        return {"error": f"HTTP {response.status}: {error_text}"}
            elif method == "POST":
                async with session.post(url, json=data, timeout=timeout) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        return {"error": f"HTTP {response.status}: {error_text}"}
        except aiohttp.ClientConnectorError:
            return {
                "error": f"Cannot connect to RL API server at {RL_API_URL}. "
                         "Make sure the server is running: "
                         "cd tinker-atropos && uvicorn rl_api_server:app --port 8080"
            }
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}


# ============================================================================
# Environment Discovery Tools
# ============================================================================

async def rl_list_environments() -> str:
    """
    List all available RL environments.
    
    Scans tinker-atropos/tinker_atropos/environments/ for Python files
    containing classes that inherit from BaseEnv.
    
    Returns information about each environment including:
    - name: Environment identifier
    - class_name: Python class name
    - file_path: Path to the environment file
    - description: Brief description if available
    
    TIP: To create or modify RL environments:
    1. Use terminal/file tools to inspect existing environments
    2. Study how they load datasets, define verifiers, and structure rewards
    3. Inspect HuggingFace datasets to understand data formats
    4. Copy an existing environment as a template
    5. Test with rl_test_inference before running full training
    
    Returns:
        JSON string with list of environments or error message
    """
    result = await _make_request("GET", "/environments")
    
    if "error" in result:
        return json.dumps(result, indent=2)
    
    # Add helpful tips to the response
    response = {
        "environments": result,
        "count": len(result),
        "tips": [
            "Use rl_select_environment(name) to select an environment",
            "Read the file_path with file tools to understand how each environment works",
            "Look for load_dataset(), score_answer(), get_next_item() methods",
        ]
    }
    
    return json.dumps(response, indent=2)


async def rl_select_environment(name: str) -> str:
    """
    Select an RL environment for training.
    
    This loads the environment's default configuration into the config state.
    After selecting, use rl_get_current_config() to see the configuration
    and rl_edit_config() to modify specific fields.
    
    Args:
        name: Name of the environment to select (from rl_list_environments)
    
    Returns:
        JSON string with selection result, file path, and current config
    
    TIP: Read the returned file_path to understand how the environment works:
    - How it loads data (load_dataset calls)
    - How it verifies answers (score_answer method)
    - What prompts it uses (system_prompt, get_next_item)
    """
    result = await _make_request("POST", f"/environments/{name}/select")
    return json.dumps(result, indent=2)


# ============================================================================
# Configuration Tools
# ============================================================================

async def rl_get_current_config() -> str:
    """
    Get the current environment configuration.
    
    Returns only the fields that are safe to modify. Other fields
    (tokenizer_name, rollout_server_url, etc.) are fixed by the system.
    
    Available fields:
    - group_size: Rollouts per prompt (4-16 typical)
    - max_token_length: Max generation tokens (2048-16384)
    - total_steps: Training steps (50-2000)
    - steps_per_eval: Steps between evaluations
    - use_wandb: Enable WandB logging
    - wandb_name: WandB run name prefix
    - max_num_workers: Concurrent workers (-1 = auto)
    
    Returns:
        JSON string with current config fields and their values
    """
    result = await _make_request("GET", "/config")
    return json.dumps(result, indent=2)


async def rl_edit_config(field: str, value: Any) -> str:
    """
    Update a configuration field.
    
    Only exposed fields can be modified. Validates field name and type.
    
    Args:
        field: Name of the field to update (e.g., "group_size", "total_steps")
        value: New value for the field
    
    Valid fields:
    - group_size (int): Rollouts per prompt
    - max_token_length (int): Max generation tokens
    - total_steps (int): Training steps
    - steps_per_eval (int): Eval frequency
    - use_wandb (bool): Enable logging
    - wandb_name (str): Run name prefix
    - max_num_workers (int): Workers count
    
    Returns:
        JSON string with updated config or error message
    """
    result = await _make_request("POST", "/config", {"field": field, "value": value})
    return json.dumps(result, indent=2)


# ============================================================================
# Training Management Tools
# ============================================================================

async def rl_start_training(
    wandb_project: str = "rl-training",
    lora_rank: int = 32,
    learning_rate: float = 4e-5,
) -> str:
    """
    Start a new RL training run with the current environment and config.
    
    Requires an environment to be selected first using rl_select_environment().
    
    WARNING: Training runs can take hours to days. Use rl_check_status() to
    monitor progress (recommended: check every 30 minutes at most).
    
    Args:
        wandb_project: WandB project name for logging
        lora_rank: LoRA rank for training (default: 32)
        learning_rate: Learning rate (default: 4e-5)
    
    Returns:
        JSON string with run_id and initial status
    
    TIP: Before starting training:
    1. Test with rl_test_inference() to verify the environment works
    2. Start with fewer total_steps to validate the setup
    3. Monitor WandB metrics for reward/mean and percent_correct
    """
    result = await _make_request("POST", "/runs", {
        "wandb_project": wandb_project,
        "lora_rank": lora_rank,
        "learning_rate": learning_rate,
    })
    return json.dumps(result, indent=2)


async def rl_check_status(run_id: str) -> str:
    """
    Get status and metrics for a training run.
    
    RATE LIMITED: For long-running training, this function enforces a
    minimum 30-minute interval between checks for the same run_id.
    
    Fetches latest metrics from WandB if available:
    - step: Current training step
    - state: Run state (running, finished, crashed)
    - reward_mean: Average reward across batches
    - loss: Training loss
    - percent_correct: Training accuracy
    - eval_percent_correct: Evaluation accuracy
    
    Args:
        run_id: The run ID returned by rl_start_training()
    
    Returns:
        JSON string with run status and metrics, or rate limit message
    """
    global _last_status_check
    
    # Check rate limiting
    now = time.time()
    if run_id in _last_status_check:
        elapsed = now - _last_status_check[run_id]
        if elapsed < MIN_STATUS_CHECK_INTERVAL:
            remaining = MIN_STATUS_CHECK_INTERVAL - elapsed
            return json.dumps({
                "rate_limited": True,
                "run_id": run_id,
                "message": f"Rate limited. Next check available in {remaining/60:.0f} minutes.",
                "next_check_in_seconds": remaining,
            }, indent=2)
    
    _last_status_check[run_id] = now
    result = await _make_request("GET", f"/runs/{run_id}")
    return json.dumps(result, indent=2)


async def rl_stop_training(run_id: str) -> str:
    """
    Stop a running training job.
    
    Use this if:
    - Metrics look bad or training is stagnant
    - You want to try different settings
    - You need to free up resources
    
    Args:
        run_id: The run ID to stop
    
    Returns:
        JSON string with stop confirmation
    """
    result = await _make_request("POST", f"/runs/{run_id}/stop")
    return json.dumps(result, indent=2)


async def rl_get_results(run_id: str) -> str:
    """
    Get final results and metrics for a completed training run.
    
    Returns:
    - Final metrics (reward, loss, accuracy)
    - WandB run URL for detailed analysis
    - Path to trained weights (tinker:// URL)
    
    Args:
        run_id: The run ID to get results for
    
    Returns:
        JSON string with final results and weights path
    """
    result = await _make_request("GET", f"/runs/{run_id}/metrics")
    return json.dumps(result, indent=2)


# ============================================================================
# Inference Testing Tools
# ============================================================================

async def rl_test_inference(
    prompts: List[str],
    max_tokens: int = 256,
    temperature: float = 1.0,
) -> str:
    """
    Test inference + verifier on sample prompts WITHOUT full training.
    
    Use this to validate environments before committing to long training runs.
    Tests:
    - Data loading and formatting
    - Model inference through Tinker
    - Verifier/reward function logic
    
    NOTE: This still requires the RL API server to be running with
    Tinker access for the Sample() method.
    
    Args:
        prompts: List of test prompts to run through the environment
        max_tokens: Maximum tokens to generate per prompt
        temperature: Sampling temperature
    
    Returns:
        JSON string with responses and verifier scores for each prompt
    
    TIP: Include prompts with known correct/incorrect answers to verify
    the reward function is working correctly.
    """
    result = await _make_request("POST", "/test/inference", {
        "prompts": prompts,
        "max_tokens": max_tokens,
        "temperature": temperature,
    })
    return json.dumps(result, indent=2)


# ============================================================================
# Utility Tools
# ============================================================================

async def rl_list_runs() -> str:
    """
    List all training runs (active and completed).
    
    Returns:
        JSON string with list of runs and their status
    """
    result = await _make_request("GET", "/runs")
    return json.dumps(result, indent=2)


# ============================================================================
# Requirements Check
# ============================================================================

def check_rl_api_keys() -> bool:
    """
    Check if required API keys are available in environment variables.
    
    Required:
    - TINKER_API_KEY: For Tinker training service
    - WANDB_API_KEY: For metrics logging and fetching
    
    Returns:
        bool: True if all required keys are set, False otherwise
    """
    tinker_key = os.getenv("TINKER_API_KEY")
    wandb_key = os.getenv("WANDB_API_KEY")
    
    return bool(tinker_key) and bool(wandb_key)


def get_missing_keys() -> List[str]:
    """
    Get list of missing required API keys.
    
    Returns:
        List of missing key names
    """
    missing = []
    if not os.getenv("TINKER_API_KEY"):
        missing.append("TINKER_API_KEY")
    if not os.getenv("WANDB_API_KEY"):
        missing.append("WANDB_API_KEY")
    return missing


# ============================================================================
# Debug/Status
# ============================================================================

async def rl_health_check() -> str:
    """
    Check if the RL API server is running and accessible.
    
    Returns:
        JSON string with server health status
    """
    result = await _make_request("GET", "/health")
    return json.dumps(result, indent=2)
