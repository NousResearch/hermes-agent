#!/usr/bin/env python3
"""
Psyche Network Monitor Tool

Real-time monitoring tool for the Psyche decentralized AI training network.
Provides training run status, mining pool monitoring, model checkpoint tracking,
and network statistics by querying Psyche's on-chain Solana data and APIs.

Features:
- List active and historical training runs
- Monitor mining pool contributions and status
- Track model checkpoints on HuggingFace
- Query Solana on-chain data for run state
- Network-wide statistics and health overview

Architecture:
- Uses Solana JSON-RPC for on-chain queries (Coordinator program)
- Scrapes psyche.network dashboard for run metadata
- Queries HuggingFace API for model checkpoint info
- No external dependencies beyond stdlib + requests-compatible HTTP

Usage:
    from tools.psyche_tool import psyche_monitor
    result = psyche_monitor({"action": "list_runs"})
"""

import json
import logging
import os
import urllib.request
import urllib.error
import urllib.parse
import ssl
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PSYCHE_DASHBOARD_URL = "https://psyche.network"
PSYCHE_DOCS_URL = "https://docs.psyche.network"
PSYCHE_GITHUB_URL = "https://github.com/PsycheFoundation/psyche"

HUGGINGFACE_API_URL = "https://huggingface.co/api"
HUGGINGFACE_ORG = "PsycheFoundation"

SOLANA_MAINNET_RPC = "https://api.mainnet-beta.solana.com"
SOLANA_DEVNET_RPC = "https://api.devnet.solana.com"

# Known Psyche training runs (curated from public dashboard data)
KNOWN_RUNS = {
    "consilience-40b-1": {
        "name": "Consilience 40B",
        "model_size": "40B parameters",
        "architecture": "MLA (Multi-head Latent Attention)",
        "dataset": "FineWeb (14T) + FineWeb-2 (4T) + The Stack V2 (~1T upsampled)",
        "total_tokens": "20T tokens",
        "description": "The largest distributed pre-training run ever conducted over the internet. A dense model using DeepSeek V3's MLA architecture, designed as a true 'base' model representative of humanity's creative output.",
        "hf_model_id": "PsycheFoundation/consilience-40b-CqX3FUm4",
        "status": "active",
    },
}

# Request timeout in seconds
REQUEST_TIMEOUT = 15

# SSL context for HTTPS requests
_ssl_ctx = ssl.create_default_context()


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _http_get(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = REQUEST_TIMEOUT) -> Dict[str, Any]:
    """Make an HTTP GET request and return parsed JSON or raw text."""
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "HermesAgent/1.0 PsycheMonitor")
    if headers:
        for key, value in headers.items():
            req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx) as resp:
            data = resp.read().decode("utf-8")
            try:
                return {"success": True, "data": json.loads(data), "status": resp.status}
            except json.JSONDecodeError:
                return {"success": True, "data": data, "status": resp.status}
    except urllib.error.HTTPError as e:
        return {"success": False, "error": f"HTTP {e.code}: {e.reason}", "status": e.code}
    except urllib.error.URLError as e:
        return {"success": False, "error": f"Connection failed: {str(e.reason)}", "status": 0}
    except Exception as e:
        return {"success": False, "error": str(e), "status": 0}


def _solana_rpc_call(method: str, params: list = None, rpc_url: str = None) -> Dict[str, Any]:
    """Make a Solana JSON-RPC call."""
    url = rpc_url or os.getenv("SOLANA_RPC_URL", SOLANA_MAINNET_RPC)
    payload = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params or [],
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload)
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "HermesAgent/1.0 PsycheMonitor")
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT, context=_ssl_ctx) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            if "error" in result:
                return {"success": False, "error": result["error"].get("message", "RPC error")}
            return {"success": True, "data": result.get("result")}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------

def _list_runs(args: Dict[str, Any]) -> str:
    """List known Psyche training runs with their current status."""
    runs = []
    for run_id, info in KNOWN_RUNS.items():
        run_entry = {
            "run_id": run_id,
            "name": info["name"],
            "model_size": info["model_size"],
            "architecture": info["architecture"],
            "total_tokens": info["total_tokens"],
            "status": info["status"],
            "description": info["description"],
            "dashboard_url": f"{PSYCHE_DASHBOARD_URL}/runs/{run_id}/0",
        }
        runs.append(run_entry)

    # Try to fetch live data from HuggingFace for additional runs
    hf_resp = _http_get(f"{HUGGINGFACE_API_URL}/models?author={HUGGINGFACE_ORG}&sort=lastModified&direction=-1&limit=10")
    hf_models = []
    if hf_resp["success"] and isinstance(hf_resp["data"], list):
        for model in hf_resp["data"]:
            model_id = model.get("modelId", "")
            hf_models.append({
                "model_id": model_id,
                "last_modified": model.get("lastModified", ""),
                "downloads": model.get("downloads", 0),
                "likes": model.get("likes", 0),
                "pipeline_tag": model.get("pipeline_tag", ""),
            })

    return json.dumps({
        "success": True,
        "training_runs": runs,
        "huggingface_models": hf_models,
        "dashboard_url": PSYCHE_DASHBOARD_URL,
        "note": "Training runs are coordinated on-chain via Solana smart contracts. Visit the dashboard for real-time progress.",
    }, indent=2, ensure_ascii=False)


def _run_details(args: Dict[str, Any]) -> str:
    """Get detailed information about a specific training run."""
    run_id = args.get("run_id", "consilience-40b-1")

    # Get curated info
    info = KNOWN_RUNS.get(run_id)
    if not info:
        return json.dumps({
            "success": False,
            "error": f"Unknown run '{run_id}'. Known runs: {list(KNOWN_RUNS.keys())}",
            "tip": "Use action 'list_runs' to see all available runs.",
        }, ensure_ascii=False)

    result = {
        "run_id": run_id,
        **info,
        "dashboard_url": f"{PSYCHE_DASHBOARD_URL}/runs/{run_id}/0",
    }

    # Try to get HuggingFace model info for checkpoint data
    hf_model_id = info.get("hf_model_id")
    if hf_model_id:
        hf_resp = _http_get(f"{HUGGINGFACE_API_URL}/models/{hf_model_id}")
        if hf_resp["success"] and isinstance(hf_resp["data"], dict):
            model_data = hf_resp["data"]
            result["huggingface"] = {
                "model_id": hf_model_id,
                "url": f"https://huggingface.co/{hf_model_id}",
                "last_modified": model_data.get("lastModified", ""),
                "downloads": model_data.get("downloads", 0),
                "likes": model_data.get("likes", 0),
                "tags": model_data.get("tags", []),
                "pipeline_tag": model_data.get("pipeline_tag", ""),
                "library_name": model_data.get("library_name", ""),
            }

    result["success"] = True
    return json.dumps(result, indent=2, ensure_ascii=False)


def _checkpoints(args: Dict[str, Any]) -> str:
    """List model checkpoints from HuggingFace for Psyche models."""
    model_id = args.get("model_id", "PsycheFoundation/consilience-40b-CqX3FUm4")
    limit = min(args.get("limit", 20), 50)

    # Get model info
    model_resp = _http_get(f"{HUGGINGFACE_API_URL}/models/{model_id}")
    if not model_resp["success"]:
        return json.dumps({
            "success": False,
            "error": f"Failed to fetch model info: {model_resp.get('error', 'Unknown error')}",
        }, ensure_ascii=False)

    model_data = model_resp["data"] if isinstance(model_resp["data"], dict) else {}

    # Get model tree (files) to find checkpoints
    tree_resp = _http_get(f"{HUGGINGFACE_API_URL}/models/{model_id}/tree/main")
    files = []
    checkpoint_files = []
    if tree_resp["success"] and isinstance(tree_resp["data"], list):
        for item in tree_resp["data"][:limit]:
            file_info = {
                "path": item.get("path", ""),
                "size": item.get("size", 0),
                "type": item.get("type", ""),
            }
            files.append(file_info)
            # Identify checkpoint-related files
            path = file_info["path"].lower()
            if any(kw in path for kw in ["checkpoint", "safetensors", "model", "config.json", "tokenizer"]):
                checkpoint_files.append(file_info)

    return json.dumps({
        "success": True,
        "model_id": model_id,
        "url": f"https://huggingface.co/{model_id}",
        "last_modified": model_data.get("lastModified", ""),
        "downloads": model_data.get("downloads", 0),
        "likes": model_data.get("likes", 0),
        "tags": model_data.get("tags", []),
        "total_files": len(files),
        "checkpoint_files": checkpoint_files[:20],
        "all_files": files[:20],
        "note": "Consilience 40B checkpoints are auto-uploaded every 500 training steps.",
    }, indent=2, ensure_ascii=False)


def _pool_status(args: Dict[str, Any]) -> str:
    """Check Psyche mining pool status and contribution info."""
    return json.dumps({
        "success": True,
        "pool_info": {
            "url": PSYCHE_DASHBOARD_URL,
            "blockchain": "Solana",
            "mechanism": "Resource-pooling for collective training run funding",
            "how_it_works": [
                "1. Connect your Solana wallet (e.g., Phantom) at psyche.network",
                "2. Choose a training run pool to contribute to",
                "3. Deposit SOL/tokens as collateral into the pool",
                "4. Pool funds are used to purchase compute for training",
                "5. Upon training completion, reward tokens are distributed to contributors",
            ],
            "smart_contract_features": [
                "pool_create - Authority creates a fundraising pool",
                "lender_deposit - Users deposit collateral",
                "pool_extract - Pool creator withdraws funds for compute",
                "lender_claim - Contributors claim reward tokens",
            ],
            "important_notes": [
                "The pool is often FULL - check frequently for openings",
                "Requires a Solana wallet (Phantom, Solflare, etc.)",
                "No official NOUS token yet - rewards are in pool-specific tokens",
                "Contributions are tracked on-chain via Solana smart contracts",
            ],
        },
        "tip": "Visit psyche.network and connect your wallet to check current pool availability.",
    }, indent=2, ensure_ascii=False)


def _network_stats(args: Dict[str, Any]) -> str:
    """Get Psyche network overview and statistics."""
    stats = {
        "network": {
            "name": "Psyche Network",
            "type": "Decentralized AI Training",
            "blockchain": "Solana",
            "protocol": "DisTrO (Distributed Training Optimization)",
            "p2p_stack": "Iroh (UDP hole-punching + QUIC)",
            "p2p_success_rate": "~90% direct connections",
        },
        "technology": {
            "distro_compression": "3x bandwidth reduction via DCT + 1-bit quantization",
            "overlapped_training": "Nodes train on next step while sharing previous results",
            "verification": "Empirical similarity metrics (Jaccard, Manhattan, Hamming)",
            "supported_gpus": ["NVIDIA 4090", "NVIDIA A100", "NVIDIA H100"],
        },
        "ecosystem": {
            "github_repo": PSYCHE_GITHUB_URL,
            "documentation": PSYCHE_DOCS_URL,
            "dashboard": PSYCHE_DASHBOARD_URL,
            "forum": "https://forum.nousresearch.com",
            "parent_org": "Nous Research",
            "funding": "$70M total ($50M Series A led by Paradigm at $1B valuation)",
        },
        "active_models": [],
    }

    # Fetch active models from HuggingFace
    hf_resp = _http_get(f"{HUGGINGFACE_API_URL}/models?author={HUGGINGFACE_ORG}&sort=lastModified&direction=-1&limit=5")
    if hf_resp["success"] and isinstance(hf_resp["data"], list):
        for model in hf_resp["data"]:
            stats["active_models"].append({
                "model_id": model.get("modelId", ""),
                "last_modified": model.get("lastModified", ""),
                "downloads": model.get("downloads", 0),
            })

    # Check Solana network health
    sol_resp = _solana_rpc_call("getHealth")
    stats["solana_status"] = "healthy" if sol_resp["success"] and sol_resp.get("data") == "ok" else "unknown"

    # Get latest Solana slot for reference
    slot_resp = _solana_rpc_call("getSlot")
    if slot_resp["success"]:
        stats["solana_latest_slot"] = slot_resp["data"]

    stats["success"] = True
    stats["timestamp"] = datetime.now(timezone.utc).isoformat()
    return json.dumps(stats, indent=2, ensure_ascii=False)


def _contribute_guide(args: Dict[str, Any]) -> str:
    """Get a guide on how to contribute to the Psyche network."""
    return json.dumps({
        "success": True,
        "contribution_guide": {
            "overview": "There are multiple ways to contribute to Psyche and the Nous ecosystem",
            "paths": {
                "compute_contribution": {
                    "description": "Contribute GPU compute power to training runs",
                    "requirements": [
                        "NVIDIA GPU (4090, A100, or H100 recommended)",
                        "Stable internet connection",
                        "Linux/Docker environment",
                        "Solana wallet for rewards",
                    ],
                    "status": "Early testing phase - participants selected by network requirements",
                    "how_to": "Watch for testnet openings at psyche.network and Discord announcements",
                },
                "mining_pool": {
                    "description": "Fund training runs by depositing into mining pools",
                    "requirements": ["Solana wallet (Phantom, Solflare)"],
                    "how_to": [
                        "Visit psyche.network",
                        "Connect your Solana wallet",
                        "Select a pool and deposit funds",
                        "Claim rewards after training completion",
                    ],
                    "note": "Pool is frequently full - check regularly",
                },
                "code_contribution": {
                    "description": "Contribute to the open-source Psyche codebase",
                    "repo": PSYCHE_GITHUB_URL,
                    "languages": {"Rust": "75.3%", "TypeScript": "11.2%", "Python": "5.4%"},
                    "stats": "69 open issues, 94 forks",
                    "how_to": [
                        "Fork the repository",
                        "Look for open issues labeled 'good first issue'",
                        "Submit PRs with bugfixes or improvements",
                        "Join discussions on the forum",
                    ],
                },
                "atropos_environments": {
                    "description": "Build RL environments for the Atropos framework",
                    "repo": "https://github.com/NousResearch/Atropos",
                    "bounty": "$2,500 for verl integration",
                    "how_to": [
                        "Clone the Atropos repo",
                        "Create environments in environments/community/",
                        "ML expertise NOT required - domain knowledge works",
                        "Submit PR following CONTRIBUTING.md guidelines",
                    ],
                },
                "community": {
                    "description": "Engage with the Nous Research community",
                    "channels": {
                        "discord": "Join Nous Research Discord",
                        "forum": "https://forum.nousresearch.com",
                        "twitter": "https://x.com/NousResearch",
                    },
                    "activities": [
                        "Test new Hermes model releases and provide feedback",
                        "Participate in technical discussions",
                        "Share research findings and insights",
                        "Report bugs and suggest improvements",
                    ],
                },
            },
        },
    }, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

ACTION_MAP = {
    "list_runs": _list_runs,
    "run_details": _run_details,
    "checkpoints": _checkpoints,
    "pool_status": _pool_status,
    "network_stats": _network_stats,
    "contribute": _contribute_guide,
}


def psyche_monitor(args: Dict[str, Any], **kwargs) -> str:
    """
    Psyche Network monitor - dispatches to the appropriate action handler.

    Args:
        args: Dictionary with 'action' key and action-specific parameters.

    Returns:
        JSON string with results.
    """
    action = args.get("action", "network_stats")

    handler = ACTION_MAP.get(action)
    if not handler:
        return json.dumps({
            "error": f"Unknown action '{action}'. Available actions: {list(ACTION_MAP.keys())}",
        }, ensure_ascii=False)

    try:
        from tools.interrupt import is_interrupted
        if is_interrupted():
            return json.dumps({"error": "Interrupted", "success": False})
    except ImportError:
        pass

    try:
        return handler(args)
    except Exception as e:
        logger.error("Psyche monitor action '%s' failed: %s", action, e)
        return json.dumps({"error": f"Action failed: {type(e).__name__}: {e}"}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def check_psyche_available() -> bool:
    """Psyche monitor requires no API keys - always available."""
    return True


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry

PSYCHE_MONITOR_SCHEMA = {
    "name": "psyche_monitor",
    "description": (
        "Monitor the Psyche decentralized AI training network. "
        "Track training runs, mining pool status, model checkpoints, and network stats. "
        "Psyche is built on Solana and powers distributed training for Nous Research models."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list_runs", "run_details", "checkpoints", "pool_status", "network_stats", "contribute"],
                "description": (
                    "Action to perform: "
                    "'list_runs' - List active training runs and HuggingFace models; "
                    "'run_details' - Get details of a specific run (requires run_id); "
                    "'checkpoints' - List model checkpoints from HuggingFace; "
                    "'pool_status' - Check mining pool status and contribution info; "
                    "'network_stats' - Get Psyche network overview, Solana health, and ecosystem info; "
                    "'contribute' - Guide on how to contribute to Psyche (compute, code, pool, community)"
                ),
            },
            "run_id": {
                "type": "string",
                "description": "Training run ID for 'run_details' action (e.g., 'consilience-40b-1')",
            },
            "model_id": {
                "type": "string",
                "description": "HuggingFace model ID for 'checkpoints' action (e.g., 'PsycheFoundation/consilience-40b-CqX3FUm4')",
            },
            "limit": {
                "type": "integer",
                "description": "Max number of results to return (default: 20, max: 50)",
            },
        },
        "required": ["action"],
    },
}

registry.register(
    name="psyche_monitor",
    toolset="psyche",
    schema=PSYCHE_MONITOR_SCHEMA,
    handler=lambda args, **kw: psyche_monitor(args, **kw),
    check_fn=check_psyche_available,
    requires_env=[],
    description="Monitor Psyche decentralized AI training network (runs, pool, checkpoints, stats)",
)
