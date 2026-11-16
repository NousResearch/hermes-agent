"""cost_budget_status — the agent's read-only view of its own spend and budget.

Reads the same cached snapshot the cron budget check uses (agent/cost_budget.py),
for the profile the agent runs in. No parameters, nothing to write. Lives in
its own non-floor ``cost_budget`` toolset: a role grants it explicitly, and a
cron job needs its owner's role to grant it (the cron RBAC ceiling).
"""
from __future__ import annotations

import logging

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

_TOP_RUNS = 5

COST_BUDGET_STATUS_SCHEMA = {
    "name": "cost_budget_status",
    "description": (
        "Show this assistant's own LLM spend today and this month, its budget and "
        "what remains, whether discretionary (proactive) jobs are being skipped for "
        "being over budget, and this month's most expensive scheduled runs. When "
        "spend_is_lower_bound is true, some tokens have no price and real spend is "
        "higher. Read-only. Check it before starting optional, expensive work."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}


def _period(spent: float, budget, fraction) -> dict:
    return {
        "spent_usd": round(spent, 4),
        "budget_usd": budget,
        "remaining_usd": round(max(0.0, budget - spent), 4) if budget else None,
        "fraction": round(fraction, 4) if fraction is not None else None,
    }


def _cost_budget_status_handler(args: dict, **_kw) -> str:
    from agent import cost_budget

    try:
        cfg = cost_budget.load_budget_config()
        configured = cfg is not None
        cfg = cfg or cost_budget.BudgetConfig()
        status = cost_budget.budget_status(cfg, cost_budget.get_spend(cfg))
    except Exception as exc:
        logger.warning("cost_budget_status failed", exc_info=True)
        return tool_error(f"Could not read spend: {exc}")
    snap = status.snap
    runs = sorted(snap.runs, key=lambda r: r.cost_usd, reverse=True)[:_TOP_RUNS]
    return tool_result(
        configured=configured,
        as_of=snap.as_of,
        today=_period(snap.day_usd, cfg.daily_usd, status.day_fraction),
        month=_period(snap.month_usd, cfg.monthly_usd, status.month_fraction),
        over_budget=status.over_budget,
        discretionary_jobs_skipped=configured and status.over_budget,
        discretionary_jobs=sorted(cfg.discretionary_jobs),
        per_run_usd=cfg.per_run_usd,
        unpriced_tokens_today=snap.day_unpriced_tokens,
        unpriced_tokens_month=snap.month_unpriced_tokens,
        unpriced_models=list(snap.unpriced_models),
        spend_is_lower_bound=snap.spend_is_lower_bound,
        costliest_runs_this_month=[
            {"job_id": r.job_id, "cost_usd": round(r.cost_usd, 4), "started_at": r.started_at}
            for r in runs
        ],
    )


registry.register(
    name="cost_budget_status",
    toolset="cost_budget",
    schema=COST_BUDGET_STATUS_SCHEMA,
    handler=lambda args, **kw: _cost_budget_status_handler(args, **kw),
    check_fn=lambda: True,
    requires_env=[],
    emoji="💰",
    max_result_size_chars=4000,
)
