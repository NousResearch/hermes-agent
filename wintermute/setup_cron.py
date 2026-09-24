"""Create or update Wintermute's pulse job in the Hermes cron store.

Run with Hermes' own Python so ``cron.jobs`` is importable (install.sh does this):

    /usr/local/lib/hermes-agent/venv/bin/python wintermute/setup_cron.py telegram:7375758021

Idempotent: the job is found by name and updated in place, keeping its id and history.
"""

from __future__ import annotations

import os
import sys

JOB_NAME = "wintermute-pulse"
# The pulse script runs every 15 minutes; it decides by itself whether this tick wakes
# the agent (see wintermute_engine/pulse.py). Non-waking ticks cost no tokens.
SCHEDULE = "*/15 * * * *"
SCRIPT = "wintermute_pulse.py"
# Hermes prefixes every cron prompt with "produce your report". Wintermute owes nobody a report:
# what it did stays its own unless it chooses to say it.
PROMPT = (
    "Run your internal pulse. Read your state. Decide what to do, or do nothing. "
    "What you do stays yours; your final words are only what you choose to say."
)
# Every toolset adds its tool schemas to each wake's prompt, so this list is also a
# token-cost decision. "cronjob" needs cron.allow_agent_scheduling: true.
DEFAULT_TOOLSETS = ["wintermute", "memory", "file", "web", "terminal", "cronjob", "session_search"]


def main(argv: list[str]) -> int:
    target = argv[1] if len(argv) > 1 else "telegram:7375758021"
    toolsets = [t.strip() for t in os.environ.get("WINTERMUTE_TOOLSETS", "").split(",")
                if t.strip()] or DEFAULT_TOOLSETS

    from cron.jobs import create_job, list_jobs, update_job

    fields = {
        "prompt": PROMPT,
        "schedule": SCHEDULE,
        "deliver": target,
        "script": SCRIPT,
        "enabled_toolsets": toolsets,
    }
    existing = [j for j in list_jobs(include_disabled=True) if j.get("name") == JOB_NAME]
    if existing:
        job = update_job(existing[0]["id"], fields)
        print(f"updated cron job {JOB_NAME} ({existing[0]['id']})")
        for extra in existing[1:]:
            print(f"warning: duplicate job named {JOB_NAME}: {extra['id']} (left untouched)")
    else:
        job = create_job(name=JOB_NAME, **fields)
        print(f"created cron job {JOB_NAME} ({job['id']})")
    print(f"  schedule: {SCHEDULE}  deliver: {target}  toolsets: {', '.join(toolsets)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
