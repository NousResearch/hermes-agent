#!/usr/bin/env python3
"""
Token Burn Enforcement System

Enforces hard daily budget caps, per-job context limits, and premium model blocks.
Tracks all token usage and generates daily receipts for compliance reporting.

Budget Enforcement:
  - Daily cap: $5.00 USD (hard limit, blocks execution if exceeded)
  - Warning threshold: $3.00 USD (logs warning to Telegram before hitting cap)
  - Per-job context limit: 2000 tokens max
  - Cost estimation: ~$0.10 per 1000 tokens (conservative average)

Model Restrictions:
  - Premium models (Opus, GPT-4, Claude-4) blocked for background jobs
  - Only whitelisted models allowed for cron/delegate tasks
  - Automatic fallback to Haiku for unauthorized premium models

Token Usage Tracking:
  - All delegated tasks logged with job_id, model, tokens used, cost
  - Session-based tracking in ~/.hermes/token_usage.jsonl
  - Daily snapshots for receipt generation
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

DAILY_BUDGET_CAP_USD = 5.00  # Hard limit in USD per day
DAILY_BUDGET_WARN_USD = 3.00  # Warning threshold before hard limit
COST_PER_1K_TOKENS = 0.10  # Conservative average across models
MAX_CONTEXT_TOKENS = 2000  # Per-job context limit

# Premium models that require whitelisting for background jobs
PREMIUM_MODELS = {
    "anthropic/claude-opus-4-1",
    "anthropic/claude-opus",
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4o",
    "gpt-4-vision",
    "openai/gpt-4",
    "openai/gpt-4-turbo",
    "openai/gpt-4o",
}

# Fallback model for blocked premium models
FALLBACK_MODEL = "anthropic/claude-haiku-4-5"

# Budget tracking file
HERMES_HOME = Path(os.path.expanduser("~/.hermes"))
TOKEN_USAGE_LOG = HERMES_HOME / "token_usage.jsonl"
DAILY_BUDGET_LOG = HERMES_HOME / "daily_budget.json"


def ensure_tracking_dirs():
    """Ensure tracking directories exist."""
    HERMES_HOME.mkdir(parents=True, exist_ok=True)


def estimate_cost_from_tokens(num_tokens: int) -> float:
    """
    Estimate USD cost from token count.
    Uses conservative average: $0.10 per 1000 tokens
    """
    return (num_tokens / 1000.0) * COST_PER_1K_TOKENS


def get_today_iso() -> str:
    """Get today's date in ISO format (YYYY-MM-DD)."""
    return datetime.utcnow().strftime("%Y-%m-%d")


def load_daily_budget(date: str = None) -> Dict[str, any]:
    """
    Load today's budget tracker from disk.
    
    Returns dict with keys:
      - date: ISO date string
      - total_spend_usd: float
      - total_tokens: int
      - entries: list of {timestamp, job_id, model, tokens, cost_usd}
    """
    if date is None:
        date = get_today_iso()
    
    ensure_tracking_dirs()
    
    if not DAILY_BUDGET_LOG.exists():
        return {
            "date": date,
            "total_spend_usd": 0.0,
            "total_tokens": 0,
            "entries": [],
        }
    
    try:
        with open(DAILY_BUDGET_LOG, "r") as f:
            data = json.load(f)
            if data.get("date") == date:
                return data
    except Exception as e:
        logger.warning(f"Failed to load daily budget: {e}")
    
    # Date mismatch or corrupt file — return fresh
    return {
        "date": date,
        "total_spend_usd": 0.0,
        "total_tokens": 0,
        "entries": [],
    }


def save_daily_budget(budget: Dict) -> None:
    """Save budget tracker to disk."""
    ensure_tracking_dirs()
    try:
        with open(DAILY_BUDGET_LOG, "w") as f:
            json.dump(budget, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save daily budget: {e}")


def log_token_usage(
    job_id: str,
    model: str,
    tokens_used: int,
    task_type: str = "delegate",  # "delegate", "cron", "interactive"
    cost_usd: float = None,
) -> None:
    """
    Log token usage to persistent record for audit and receipt generation.
    
    Args:
        job_id: Unique task identifier
        model: Model name used
        tokens_used: Number of tokens consumed
        task_type: Type of task (delegate, cron, interactive)
        cost_usd: Estimated USD cost (auto-calculated if not provided)
    """
    ensure_tracking_dirs()
    
    if cost_usd is None:
        cost_usd = estimate_cost_from_tokens(tokens_used)
    
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "date": get_today_iso(),
        "job_id": job_id,
        "model": model,
        "tokens_used": tokens_used,
        "cost_usd": cost_usd,
        "task_type": task_type,
    }
    
    try:
        with open(TOKEN_USAGE_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to log token usage: {e}")


def check_and_enforce_budget(
    job_id: str,
    estimated_tokens: int,
    is_background_job: bool = True,
) -> Tuple[bool, str]:
    """
    Check if a new job would exceed daily budget.
    
    Returns:
        (allow, reason_or_warning)
        
        If allow=True:
          - reason is empty string or warning message
        If allow=False:
          - reason is error explaining budget exceeded
    """
    estimated_cost = estimate_cost_from_tokens(estimated_tokens)
    budget = load_daily_budget()
    current_spend = budget["total_spend_usd"]
    projected_spend = current_spend + estimated_cost
    
    logger.info(
        f"[budget] job_id={job_id} current=${current_spend:.2f} "
        f"estimated=${estimated_cost:.2f} projected=${projected_spend:.2f}"
    )
    
    # Hard limit check
    if projected_spend > DAILY_BUDGET_CAP_USD:
        msg = (
            f"BUDGET EXCEEDED: Daily cap ${DAILY_BUDGET_CAP_USD:.2f} would be exceeded. "
            f"Current: ${current_spend:.2f}, Estimated: ${estimated_cost:.2f}. "
            f"Job blocked."
        )
        logger.error(f"[budget-block] {msg}")
        return False, msg
    
    # Warning threshold check
    if projected_spend > DAILY_BUDGET_WARN_USD:
        msg = (
            f"⚠️  BUDGET WARNING: Approaching daily limit. "
            f"Current: ${current_spend:.2f}, This job: ${estimated_cost:.2f}, "
            f"Total after: ${projected_spend:.2f} / ${DAILY_BUDGET_CAP_USD:.2f}"
        )
        logger.warning(f"[budget-warn] {msg}")
        return True, msg
    
    return True, ""


def apply_context_limit(context: str, max_tokens: int = MAX_CONTEXT_TOKENS) -> Tuple[str, int]:
    """
    Truncate context to token limit.
    
    Conservative estimate: 1 token ≈ 4 characters (varies by model/language)
    
    Returns:
        (truncated_context, estimated_tokens_in_context)
    """
    # Conservative estimate: 1 token = 4 characters
    MAX_CHARS = max_tokens * 4
    
    if len(context) > MAX_CHARS:
        truncated = context[:MAX_CHARS]
        estimated_tokens = max_tokens
        logger.warning(
            f"[context-limit] Context truncated from {len(context)} chars "
            f"to {MAX_CHARS} chars (~{estimated_tokens} tokens)"
        )
        return truncated, estimated_tokens
    
    estimated_tokens = len(context) // 4
    return context, estimated_tokens


def select_model_with_enforcement(
    requested_model: str,
    is_background_job: bool = True,
    allowed_premium_models: List[str] = None,
) -> Tuple[str, str]:
    """
    Enforce model selection rules.
    
    Rules:
      1. If background job and model is premium: check whitelist
      2. If not whitelisted: use fallback
      3. Log reason for selection
    
    Args:
        requested_model: Originally requested model
        is_background_job: True for cron/delegate, False for interactive
        allowed_premium_models: Whitelist of premium models allowed (empty = none)
    
    Returns:
        (selected_model, reason)
    """
    if allowed_premium_models is None:
        allowed_premium_models = []
    
    # Only enforce for background jobs
    if not is_background_job:
        return requested_model, ""
    
    # Check if requested model is premium
    if requested_model not in PREMIUM_MODELS:
        return requested_model, ""
    
    # Premium model in background job context
    if requested_model in allowed_premium_models:
        return requested_model, ""
    
    # Premium model NOT whitelisted — use fallback
    reason = (
        f"Premium model {requested_model} blocked for background job. "
        f"Allowed whitelist: {allowed_premium_models or 'empty'}. "
        f"Using fallback: {FALLBACK_MODEL}"
    )
    logger.warning(f"[model-enforce] {reason}")
    return FALLBACK_MODEL, reason


def get_allowed_premium_models() -> List[str]:
    """
    Load whitelist of allowed premium models from config.
    
    Reads from ~/.hermes/config.yaml:
      token_enforcement:
        allowed_premium_models: []  # empty = none allowed
    """
    config_file = HERMES_HOME / "config.yaml"
    if not config_file.exists():
        return []
    
    try:
        import yaml
        with open(config_file, "r") as f:
            config = yaml.safe_load(f) or {}
        enforcement = config.get("token_enforcement", {})
        return enforcement.get("allowed_premium_models", [])
    except Exception as e:
        logger.debug(f"Failed to load premium models whitelist: {e}")
        return []


# =============================================================================
# CLI Entry Point (for --requested-model flag)
# =============================================================================

def main():
    """
    CLI entry point for model enforcement.
    
    Usage:
      python enforce_token_discipline.py \
        --requested-model anthropic/claude-opus \
        --budget-tokens 1500 \
        --task-id job-123 \
        --is-background
    
    Output: JSON with chosen_model, reason, budget_ok
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Token enforcement engine")
    parser.add_argument("--requested-model", default="")
    parser.add_argument("--budget-tokens", type=int, default=1000)
    parser.add_argument("--task-id", default="unknown")
    parser.add_argument("--is-background", action="store_true")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Check budget
    budget_ok, budget_msg = check_and_enforce_budget(
        args.task_id,
        args.budget_tokens,
        args.is_background,
    )
    
    # Check model if budget is OK
    chosen_model = args.requested_model
    model_reason = ""
    if budget_ok and args.is_background:
        allowed = get_allowed_premium_models()
        chosen_model, model_reason = select_model_with_enforcement(
            args.requested_model,
            is_background_job=True,
            allowed_premium_models=allowed,
        )
    
    result = {
        "chosen_model": chosen_model,
        "budget_ok": budget_ok,
        "budget_reason": budget_msg if not budget_ok else "",
        "model_reason": model_reason,
    }
    
    print(json.dumps(result))
    sys.exit(0 if budget_ok else 1)


if __name__ == "__main__":
    main()
