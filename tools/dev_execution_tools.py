"""Hermes tools that let Dev create and launch worker execution plans."""

from __future__ import annotations

from typing import Any, Dict

from gateway.dev_execution import (
    DevExecutionStore,
    apply_execution_plan_review,
    apply_supervisor_approval,
    approve_supervisor_approval,
    deny_supervisor_approval,
    derive_execution_plan_status,
    get_runbook,
    get_supervisor_approval,
    launch_execution_plan,
    list_supervisor_loop_status,
    list_runbooks,
    list_supervisor_approvals,
    list_launch_profiles,
    list_worker_runtimes,
    next_execution_step,
    review_execution_plan,
    select_execution_runtime,
    set_project_runbook,
    set_supervisor_loop,
    set_execution_plan_test_state,
    supervise_execution_plans,
    synthesize_execution_plan,
)
from gateway.dev_control.harness_observability import (
    generate_harness_report,
    list_harness_components,
)
from gateway.dev_control.harness_benchmarks import (
    get_harness_benchmark_run,
    list_harness_benchmark_runs,
    run_harness_benchmark,
)
from gateway.dev_control.harness_recommendations import (
    generate_harness_recommendations,
    get_harness_recommendation_run,
    list_harness_recommendation_runs,
)
from gateway.dev_control.clarifications import (
    DevClarificationStore,
    answer_clarification,
    cancel_clarification,
    complete_clarification,
    get_clarification,
    list_clarifications,
    start_clarification,
)
from gateway.dev_control.plan_artifacts import (
    DevPlanArtifactStore,
    approve_execution_plan_draft,
    approve_plan_artifact,
    cancel_execution_plan_draft,
    cancel_plan_artifact,
    create_execution_plan_from_artifact,
    create_plan_artifact,
    get_execution_plan_draft_review,
    get_plan_artifact,
    list_plan_artifact_builds,
    list_plan_artifacts,
    revise_execution_plan_draft,
    revise_plan_artifact,
)
from gateway.subagent_events import SubagentEventStore
from tools.openhands_bridge import (
    openhands_server_status,
    start_openhands_server,
    stop_openhands_server,
)
from tools.registry import registry, tool_error, tool_result


DEV_LAUNCH_PROFILES_SCHEMA = {
    "name": "dev_launch_profiles",
    "description": "List Hermes-owned Dev launch profiles for AO worker delegation.",
    "parameters": {"type": "object", "properties": {}},
}


DEV_WORKER_RUNTIMES_SCHEMA = {
    "name": "dev_worker_runtimes",
    "description": "List Hermes Dev worker runtimes and their launch/action capabilities.",
    "parameters": {"type": "object", "properties": {}},
}


DEV_SELECT_WORKER_RUNTIME_SCHEMA = {
    "name": "dev_select_worker_runtime",
    "description": "Dry-run Hermes Dev runtime selection for a task without creating or launching a plan.",
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {"type": "string", "description": "Task goal."},
            "prompt": {"type": "string", "description": "Task prompt or brief."},
            "profile_id": {"type": "string", "description": "Optional Dev launch profile id."},
            "runtime": {"type": "string", "description": "Requested runtime, including auto."},
            "project_id": {"type": "string", "description": "Project id."},
            "permissions": {"type": "string", "description": "Task permissions, such as read_only or edit."},
        },
    },
}


DEV_HARNESS_COMPONENTS_SCHEMA = {
    "name": "dev_harness_components",
    "description": "List active Hermes Dev harness components and version hashes.",
    "parameters": {"type": "object", "properties": {}},
}


DEV_HARNESS_REPORT_SCHEMA = {
    "name": "dev_harness_report",
    "description": "Generate an observe-only Dev harness experience report from persisted plans and subagent events.",
    "parameters": {
        "type": "object",
        "properties": {
            "plan_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional plan ids to include. Defaults to recent plans.",
            },
            "project_id": {"type": "string", "description": "Optional project id filter."},
            "limit": {"type": "integer", "description": "Maximum number of recent plans when plan_ids is omitted."},
            "since": {"type": "number", "description": "Optional minimum plan updated_at timestamp."},
            "persist": {"type": "boolean", "default": True},
        },
    },
}


DEV_HARNESS_RECOMMENDATIONS_SCHEMA = {
    "name": "dev_harness_recommendations",
    "description": "Generate persisted, recommendation-only Dev harness improvement guidance.",
    "parameters": {
        "type": "object",
        "properties": {
            "report_id": {"type": "string", "description": "Existing harness report id to analyze."},
            "plan_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional plan ids for generating a fresh source report.",
            },
            "project_id": {"type": "string", "description": "Optional project id filter for a generated report."},
            "limit": {"type": "integer", "description": "Maximum recent plans for a generated report."},
            "since": {"type": "number", "description": "Optional minimum plan updated_at timestamp."},
            "benchmark_run_id": {"type": "string", "description": "Optional benchmark run id to reference."},
            "persist": {"type": "boolean", "default": True},
        },
    },
}


DEV_HARNESS_RECOMMENDATION_RUNS_SCHEMA = {
    "name": "dev_harness_recommendation_runs",
    "description": "List or fetch persisted Dev harness recommendation runs.",
    "parameters": {
        "type": "object",
        "properties": {
            "recommendation_run_id": {"type": "string", "description": "Optional recommendation run id to fetch."},
            "report_id": {"type": "string", "description": "Optional source report id filter for listing."},
            "limit": {"type": "integer", "description": "Maximum recommendation runs to list."},
        },
    },
}


DEV_HARNESS_BENCHMARK_SCHEMA = {
    "name": "dev_harness_benchmark",
    "description": "Run a controlled Dev runtime benchmark. Real workers launch only when live=true.",
    "parameters": {
        "type": "object",
        "properties": {
            "runtimes": {"type": "array", "items": {"type": "string"}, "description": "Runtime ids to benchmark."},
            "cases": {"type": "array", "items": {"type": "object"}, "description": "Optional benchmark case specs."},
            "mode": {"type": "string", "description": "dry_run or fixture. live mode requires live=true."},
            "live": {"type": "boolean", "default": False},
            "project_id": {"type": "string", "description": "Project id. Defaults to OrynWorkspace."},
            "max_cases": {"type": "integer", "description": "Maximum benchmark cases. Defaults to 3."},
            "iterations": {"type": "integer", "description": "Iterations per runtime/case pair. Defaults to 1; live mode is capped at 3."},
            "timeout_seconds": {"type": "integer", "description": "Live polling timeout. Defaults to 180."},
            "persist": {"type": "boolean", "default": True},
        },
    },
}


DEV_HARNESS_BENCHMARK_RUNS_SCHEMA = {
    "name": "dev_harness_benchmark_runs",
    "description": "List or fetch persisted Dev harness benchmark runs.",
    "parameters": {
        "type": "object",
        "properties": {
            "benchmark_run_id": {"type": "string", "description": "Optional benchmark run id to fetch."},
            "limit": {"type": "integer", "description": "Maximum benchmark runs to list."},
        },
    },
}


DEV_START_CLARIFICATION_SCHEMA = {
    "name": "dev_start_clarification",
    "description": "Start a durable Dev planning clarification session from a rough vision brief.",
    "parameters": {
        "type": "object",
        "properties": {
            "vision_brief": {"type": "string", "description": "Rough vision or feature brief to clarify."},
            "project_id": {"type": "string", "description": "Project id. Defaults to OrynWorkspace."},
            "session_id": {"type": "string", "description": "Optional Oryn/Hermes chat session id to associate."},
            "project_context": {"type": "object", "description": "Optional project context such as name, vision, repositories, and work items."},
            "max_questions": {"type": "integer", "description": "Maximum questions, capped at 5."},
        },
        "required": ["vision_brief"],
    },
}


DEV_CLARIFICATIONS_SCHEMA = {
    "name": "dev_clarifications",
    "description": "List or fetch durable Dev planning clarification sessions.",
    "parameters": {
        "type": "object",
        "properties": {
            "clarification_id": {"type": "string", "description": "Optional clarification id to fetch."},
            "project_id": {"type": "string", "description": "Optional project id filter."},
            "session_id": {"type": "string", "description": "Optional session id filter."},
            "status": {"type": "string", "description": "Optional status filter."},
            "limit": {"type": "integer", "description": "Maximum sessions to list."},
        },
    },
}


DEV_ANSWER_CLARIFICATION_SCHEMA = {
    "name": "dev_answer_clarification",
    "description": "Answer, skip, or go back in an active Dev clarification session.",
    "parameters": {
        "type": "object",
        "properties": {
            "clarification_id": {"type": "string"},
            "question_id": {"type": "string"},
            "option_id": {"type": "string"},
            "answer_text": {"type": "string"},
            "skipped": {"type": "boolean", "default": False},
            "back": {"type": "boolean", "default": False},
        },
        "required": ["clarification_id"],
    },
}


DEV_COMPLETE_CLARIFICATION_SCHEMA = {
    "name": "dev_complete_clarification",
    "description": "Complete a Dev clarification session and produce a structured clarified brief.",
    "parameters": {
        "type": "object",
        "properties": {
            "clarification_id": {"type": "string"},
        },
        "required": ["clarification_id"],
    },
}


DEV_CANCEL_CLARIFICATION_SCHEMA = {
    "name": "dev_cancel_clarification",
    "description": "Cancel an active Dev clarification session.",
    "parameters": {
        "type": "object",
        "properties": {
            "clarification_id": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["clarification_id"],
    },
}


DEV_PLAN_ARTIFACTS_SCHEMA = {
    "name": "dev_plan_artifacts",
    "description": "List or fetch durable Dev planning artifacts generated from clarifications.",
    "parameters": {
        "type": "object",
        "properties": {
            "plan_artifact_id": {"type": "string", "description": "Optional artifact id to fetch."},
            "clarification_id": {"type": "string", "description": "Optional clarification id filter."},
            "project_id": {"type": "string", "description": "Optional project id filter."},
            "status": {"type": "string", "description": "Optional artifact status filter."},
            "limit": {"type": "integer", "description": "Maximum artifacts to list."},
        },
    },
}


DEV_CREATE_PLAN_ARTIFACT_SCHEMA = {
    "name": "dev_create_plan_artifact",
    "description": "Create a durable planning artifact from a completed Dev clarification session.",
    "parameters": {
        "type": "object",
        "properties": {
            "clarification_id": {"type": "string"},
        },
        "required": ["clarification_id"],
    },
}


DEV_REVISE_PLAN_ARTIFACT_SCHEMA = {
    "name": "dev_revise_plan_artifact",
    "description": "Create a revised version of a durable Dev planning artifact from feedback.",
    "parameters": {
        "type": "object",
        "properties": {
            "plan_artifact_id": {"type": "string"},
            "feedback_instruction": {"type": "string"},
            "feedback": {"type": "string"},
        },
        "required": ["plan_artifact_id"],
    },
}


DEV_APPROVE_PLAN_ARTIFACT_SCHEMA = {
    "name": "dev_approve_plan_artifact",
    "description": "Mark a Dev planning artifact approved as ready for a future build transition.",
    "parameters": {
        "type": "object",
        "properties": {
            "plan_artifact_id": {"type": "string"},
        },
        "required": ["plan_artifact_id"],
    },
}


DEV_CANCEL_PLAN_ARTIFACT_SCHEMA = {
    "name": "dev_cancel_plan_artifact",
    "description": "Cancel a Dev planning artifact without deleting its history.",
    "parameters": {
        "type": "object",
        "properties": {
            "plan_artifact_id": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["plan_artifact_id"],
    },
}


DEV_CREATE_EXECUTION_PLAN_FROM_ARTIFACT_SCHEMA = {
    "name": "dev_create_execution_plan_from_artifact",
    "description": "Convert an approved Dev planning artifact into a planned Dev execution plan without launching workers.",
    "parameters": {
        "type": "object",
        "properties": {
            "plan_artifact_id": {"type": "string"},
        },
        "required": ["plan_artifact_id"],
    },
}


DEV_PLAN_ARTIFACT_BUILDS_SCHEMA = {
    "name": "dev_plan_artifact_builds",
    "description": "List build audit records linking a Dev planning artifact to draft Dev execution plans.",
    "parameters": {
        "type": "object",
        "properties": {
            "plan_artifact_id": {"type": "string"},
            "limit": {"type": "integer", "description": "Maximum build records to list."},
        },
        "required": ["plan_artifact_id"],
    },
}


DEV_EXECUTION_PLAN_DRAFT_REVIEW_SCHEMA = {
    "name": "dev_execution_plan_draft_review",
    "description": "Get draft-review metadata for an artifact-created Dev execution plan.",
    "parameters": {
        "type": "object",
        "properties": {"plan_id": {"type": "string"}},
        "required": ["plan_id"],
    },
}


DEV_REVISE_EXECUTION_PLAN_DRAFT_SCHEMA = {
    "name": "dev_revise_execution_plan_draft",
    "description": "Regenerate an unlaunched artifact-created Dev execution plan draft from feedback.",
    "parameters": {
        "type": "object",
        "properties": {
            "plan_id": {"type": "string"},
            "feedback_instruction": {"type": "string"},
            "feedback": {"type": "string"},
        },
        "required": ["plan_id"],
    },
}


DEV_APPROVE_EXECUTION_PLAN_DRAFT_SCHEMA = {
    "name": "dev_approve_execution_plan_draft",
    "description": "Mark an artifact-created Dev execution plan draft as approved for later launch.",
    "parameters": {
        "type": "object",
        "properties": {"plan_id": {"type": "string"}},
        "required": ["plan_id"],
    },
}


DEV_CANCEL_EXECUTION_PLAN_DRAFT_SCHEMA = {
    "name": "dev_cancel_execution_plan_draft",
    "description": "Cancel an artifact-created Dev execution plan draft without deleting history.",
    "parameters": {
        "type": "object",
        "properties": {
            "plan_id": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["plan_id"],
    },
}


DEV_OPENHANDS_SERVER_STATUS_SCHEMA = {
    "name": "dev_openhands_server_status",
    "description": "Inspect the local dev OpenHands server used by Hermes runtime routing.",
    "parameters": {"type": "object", "properties": {}},
}


DEV_START_OPENHANDS_SERVER_SCHEMA = {
    "name": "dev_start_openhands_server",
    "description": "Start the local dev OpenHands server if the OpenHands CLI is installed.",
    "parameters": {
        "type": "object",
        "properties": {
            "cwd": {
                "type": "string",
                "description": "Workspace directory to run openhands serve --mount-cwd from. Defaults to the Oryn repo root.",
            },
            "server_url": {
                "type": "string",
                "description": "Expected server URL. Defaults to http://127.0.0.1:3000.",
            },
            "wait_seconds": {
                "type": "number",
                "description": "Seconds to wait for /health after starting. Defaults to 5.",
            },
        },
    },
}


DEV_STOP_OPENHANDS_SERVER_SCHEMA = {
    "name": "dev_stop_openhands_server",
    "description": "Stop the local dev OpenHands server started by Hermes.",
    "parameters": {"type": "object", "properties": {}},
}


DEV_CREATE_EXECUTION_PLAN_SCHEMA = {
    "name": "dev_create_execution_plan",
    "description": "Persist a Dev execution plan without spawning workers.",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Short plan title."},
            "vision_brief": {"type": "string", "description": "Product/vision context for the plan."},
            "runbook_id": {"type": "string", "description": "Optional Dev runbook override for this plan."},
            "policy_profile": {"type": "string", "description": "Optional policy profile override for this plan."},
            "tasks": {
                "type": "array",
                "minItems": 1,
                "description": "Planned Dev worker task specs.",
                "items": {
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string"},
                        "prompt": {"type": "string"},
                        "profile_id": {"type": "string"},
                        "runtime": {"type": "string"},
                        "project_id": {"type": "string"},
                        "permissions": {
                            "type": "string",
                            "description": "Task permission level, such as read_only or edit.",
                        },
                        "agent": {"type": "string"},
                        "model": {"type": "string"},
                        "reasoning_effort": {"type": "string"},
                        "dependencies": {"type": "array", "items": {"type": "string"}},
                        "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
                        "issue_id": {"type": "string"},
                        "branch": {"type": "string"},
                    },
                    "required": ["prompt"],
                },
            },
        },
        "required": ["title", "tasks"],
    },
}


DEV_LAUNCH_EXECUTION_PLAN_SCHEMA = {
    "name": "dev_launch_execution_plan",
    "description": "Launch runnable AO workers from a persisted Dev execution plan.",
    "parameters": {
        "type": "object",
        "properties": {
            "plan_id": {"type": "string", "description": "Dev execution plan id."},
            "task_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional subset of task ids to launch.",
            },
        },
        "required": ["plan_id"],
    },
}


DEV_EXECUTION_PLAN_STATUS_SCHEMA = {
    "name": "dev_execution_plan_status",
    "description": "Return Hermes-derived Dev execution plan/task status from linked AO sessions.",
    "parameters": {
        "type": "object",
        "properties": {
            "plan_id": {"type": "string", "description": "Dev execution plan id."},
        },
        "required": ["plan_id"],
    },
}


DEV_SYNTHESIZE_EXECUTION_PLAN_SCHEMA = {
    "name": "dev_synthesize_execution_plan",
    "description": "Synthesize a compact implementation report for a Dev execution plan.",
    "parameters": {
        "type": "object",
        "properties": {
            "plan_id": {"type": "string", "description": "Dev execution plan id."},
        },
        "required": ["plan_id"],
    },
}


DEV_REVIEW_EXECUTION_PLAN_SCHEMA = {
    "name": "dev_review_execution_plan",
    "description": "Return a recommendation-only Dev review decision for an execution plan.",
    "parameters": {
        "type": "object",
        "properties": {
            "plan_id": {"type": "string", "description": "Dev execution plan id."},
            "include_synthesis": {
                "type": "boolean",
                "description": "Whether to include the compact synthesis report in the response.",
                "default": True,
            },
        },
        "required": ["plan_id"],
    },
}


DEV_APPLY_EXECUTION_PLAN_REVIEW_SCHEMA = {
    "name": "dev_apply_execution_plan_review",
    "description": "Apply the current Hermes review recommendation for a Dev execution plan.",
    "parameters": {
        "type": "object",
        "properties": {
            "plan_id": {"type": "string", "description": "Dev execution plan id."},
            "include_synthesis": {
                "type": "boolean",
                "description": "Whether to include synthesis in the re-derived review payload.",
                "default": True,
            },
            "message": {
                "type": "string",
                "description": "Optional follow-up message override when the review recommends follow_up.",
            },
            "instruction": {
                "type": "string",
                "description": "Optional retry/repair/reassign instruction override.",
            },
            "project_id": {
                "type": "string",
                "description": "Optional project override for reassign.",
            },
            "agent": {
                "type": "string",
                "description": "Optional coding agent override for reassign.",
            },
            "model": {
                "type": "string",
                "description": "Optional model override for reassign.",
            },
            "reasoning_effort": {
                "type": "string",
                "description": "Optional reasoning effort override for reassign.",
            },
        },
        "required": ["plan_id"],
    },
}


DEV_SUPERVISE_EXECUTION_PLANS_SCHEMA = {
    "name": "dev_supervise_execution_plans",
    "description": "Audit Dev execution plans and apply guarded low-risk review actions.",
    "parameters": {
        "type": "object",
        "properties": {
            "plan_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional explicit Dev execution plan ids to supervise.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum recent plans to inspect when plan_ids is omitted.",
                "default": 20,
            },
            "project_id": {
                "type": "string",
                "description": "Optional project id filter when plan_ids is omitted.",
            },
            "reviewable_only": {
                "type": "boolean",
                "description": "Only inspect terminal/reviewable plans when plan_ids is omitted.",
                "default": False,
            },
            "apply_guarded_actions": {
                "type": "boolean",
                "description": "Whether to auto-apply guarded accept/follow-up actions.",
                "default": True,
            },
            "include_synthesis": {
                "type": "boolean",
                "description": "Whether each review should include synthesis.",
                "default": False,
            },
        },
    },
}


DEV_RUNBOOKS_SCHEMA = {
    "name": "dev_runbooks",
    "description": "List or fetch Dev runbooks and built-in policy profiles.",
    "parameters": {
        "type": "object",
        "properties": {
            "runbook_id": {"type": "string", "description": "Optional runbook id to fetch."},
            "project_id": {"type": "string", "description": "Optional project id filter."},
            "limit": {"type": "integer", "description": "Maximum runbooks to list.", "default": 100},
        },
    },
}


DEV_SET_PROJECT_RUNBOOK_SCHEMA = {
    "name": "dev_set_project_runbook",
    "description": "Create or update the default Dev supervisor runbook for a project.",
    "parameters": {
        "type": "object",
        "properties": {
            "project_id": {"type": "string"},
            "policy_profile": {
                "type": "string",
                "description": "conservative, standard, or aggressive.",
                "default": "standard",
            },
            "max_follow_ups_per_task": {"type": "integer"},
            "max_retries_per_task": {"type": "integer"},
            "supervisor_enabled": {"type": "boolean", "description": "Whether Hermes should supervise this project in the background."},
            "supervisor_interval_seconds": {"type": "integer", "description": "Background supervisor cadence. Minimum 15 seconds."},
            "supervisor_limit": {"type": "integer", "description": "Maximum recent reviewable plans per tick."},
            "supervisor_include_synthesis": {"type": "boolean"},
            "supervisor_apply_guarded_actions": {"type": "boolean"},
        },
        "required": ["project_id"],
    },
}


DEV_SUPERVISOR_LOOP_STATUS_SCHEMA = {
    "name": "dev_supervisor_loop_status",
    "description": "Inspect project-opt-in Dev supervisor loop state.",
    "parameters": {
        "type": "object",
        "properties": {
            "project_id": {"type": "string", "description": "Optional project id filter."},
        },
    },
}


DEV_SET_SUPERVISOR_LOOP_SCHEMA = {
    "name": "dev_set_supervisor_loop",
    "description": "Enable or configure the guarded Dev supervisor loop for one project.",
    "parameters": {
        "type": "object",
        "properties": {
            "project_id": {"type": "string"},
            "supervisor_enabled": {"type": "boolean"},
            "supervisor_interval_seconds": {"type": "integer", "default": 60},
            "supervisor_limit": {"type": "integer", "default": 10},
            "supervisor_include_synthesis": {"type": "boolean", "default": False},
            "supervisor_apply_guarded_actions": {"type": "boolean", "default": True},
        },
        "required": ["project_id"],
    },
}


DEV_NEXT_EXECUTION_STEP_SCHEMA = {
    "name": "dev_next_execution_step",
    "description": "Return the next recommended Dev action for one execution plan without mutating state.",
    "parameters": {
        "type": "object",
        "properties": {
            "plan_id": {"type": "string", "description": "Dev execution plan id."},
            "include_synthesis": {
                "type": "boolean",
                "description": "Whether to include synthesis inside the review payload.",
                "default": False,
            },
        },
        "required": ["plan_id"],
    },
}


DEV_SET_EXECUTION_PLAN_TEST_STATE_SCHEMA = {
    "name": "dev_set_execution_plan_test_state",
    "description": "Inject deterministic normalized fixture state for a Dev plan task without calling AO.",
    "parameters": {
        "type": "object",
        "properties": {
            "plan_id": {"type": "string"},
            "task_id": {"type": "string"},
            "state": {
                "type": "string",
                "description": "completed_ok, completed_weak, failed_repairable, failed_unrepairable, or running.",
            },
            "summary": {"type": "string"},
            "status_reason": {"type": "string"},
            "ao_session_id": {"type": "string"},
            "project_id": {"type": "string"},
            "files_read": {"type": "array", "items": {"type": "string"}},
            "files_written": {"type": "array", "items": {"type": "string"}},
            "verification_evidence": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["plan_id", "task_id", "state"],
    },
}


DEV_SUPERVISOR_APPROVALS_SCHEMA = {
    "name": "dev_supervisor_approvals",
    "description": "List or fetch durable Dev supervisor approval requests.",
    "parameters": {
        "type": "object",
        "properties": {
            "approval_id": {"type": "string", "description": "Optional approval id to fetch."},
            "plan_id": {"type": "string", "description": "Optional plan id filter."},
            "status": {"type": "string", "description": "Optional approval status filter."},
            "limit": {"type": "integer", "description": "Maximum approvals to list.", "default": 50},
        },
    },
}


DEV_APPROVE_SUPERVISOR_ACTION_SCHEMA = {
    "name": "dev_approve_supervisor_action",
    "description": "Approve one pending Dev supervisor retry-family action request.",
    "parameters": {
        "type": "object",
        "properties": {
            "approval_id": {"type": "string"},
            "message": {"type": "string"},
            "instruction": {"type": "string"},
            "project_id": {"type": "string"},
            "agent": {"type": "string"},
            "model": {"type": "string"},
            "reasoning_effort": {"type": "string"},
        },
        "required": ["approval_id"],
    },
}


DEV_DENY_SUPERVISOR_ACTION_SCHEMA = {
    "name": "dev_deny_supervisor_action",
    "description": "Deny one pending Dev supervisor approval request.",
    "parameters": {
        "type": "object",
        "properties": {
            "approval_id": {"type": "string"},
            "message": {"type": "string"},
        },
        "required": ["approval_id"],
    },
}


DEV_APPLY_SUPERVISOR_APPROVAL_SCHEMA = {
    "name": "dev_apply_supervisor_approval",
    "description": "Apply one approved Dev supervisor approval request exactly once.",
    "parameters": {
        "type": "object",
        "properties": {
            "approval_id": {"type": "string"},
            "include_synthesis": {"type": "boolean", "default": True},
        },
        "required": ["approval_id"],
    },
}


def _handle_dev_launch_profiles(args: Dict[str, Any], **kwargs) -> str:
    return tool_result({"ok": True, "object": "list", "data": list_launch_profiles()})


def _handle_dev_worker_runtimes(args: Dict[str, Any], **kwargs) -> str:
    return tool_result({"ok": True, "object": "list", "data": list_worker_runtimes()})


def _handle_dev_select_worker_runtime(args: Dict[str, Any], **kwargs) -> str:
    try:
        result = select_execution_runtime(
            goal=args.get("goal"),
            prompt=args.get("prompt"),
            profile_id=args.get("profile_id") or args.get("launch_profile_id"),
            runtime=args.get("runtime"),
            project_id=args.get("project_id"),
            permissions=args.get("permissions"),
        )
    except Exception as exc:
        return tool_error(f"Dev runtime selection failed: {exc}")
    return tool_result({"ok": True, "object": "hermes.dev_runtime_selection", **result})


def _handle_dev_harness_components(args: Dict[str, Any], **kwargs) -> str:
    try:
        store = DevExecutionStore()
        components = list_harness_components(store=store)
    except Exception as exc:
        return tool_error(f"Dev harness components failed: {exc}")
    return tool_result({"ok": True, "object": "list", "data": components, "total": len(components)})


def _handle_dev_harness_report(args: Dict[str, Any], **kwargs) -> str:
    try:
        report = generate_harness_report(
            store=DevExecutionStore(),
            event_store=SubagentEventStore(),
            plan_ids=args.get("plan_ids"),
            project_id=args.get("project_id"),
            limit=args.get("limit") or 25,
            since=args.get("since"),
            persist=bool(args.get("persist", True)),
        )
    except KeyError as exc:
        return tool_error(str(exc))
    except Exception as exc:
        return tool_error(f"Dev harness report failed: {exc}")
    return tool_result(report)


def _handle_dev_harness_recommendations(args: Dict[str, Any], **kwargs) -> str:
    try:
        result = generate_harness_recommendations(
            store=DevExecutionStore(),
            event_store=SubagentEventStore(),
            report_id=args.get("report_id"),
            plan_ids=args.get("plan_ids"),
            project_id=args.get("project_id"),
            limit=args.get("limit") or 25,
            since=args.get("since"),
            benchmark_run_id=args.get("benchmark_run_id"),
            persist=bool(args.get("persist", True)),
        )
    except KeyError as exc:
        return tool_error(str(exc))
    except Exception as exc:
        return tool_error(f"Dev harness recommendations failed: {exc}")
    return tool_result(result)


def _handle_dev_harness_recommendation_runs(args: Dict[str, Any], **kwargs) -> str:
    try:
        store = DevExecutionStore()
        recommendation_run_id = str(args.get("recommendation_run_id") or "").strip()
        if recommendation_run_id:
            result = get_harness_recommendation_run(
                store=store,
                recommendation_run_id=recommendation_run_id,
            )
        else:
            result = list_harness_recommendation_runs(
                store=store,
                report_id=args.get("report_id"),
                limit=args.get("limit") or 50,
            )
    except KeyError as exc:
        return tool_error(str(exc))
    except Exception as exc:
        return tool_error(f"Dev harness recommendation lookup failed: {exc}")
    return tool_result(result)


def _handle_dev_harness_benchmark(args: Dict[str, Any], **kwargs) -> str:
    try:
        result = run_harness_benchmark(
            store=DevExecutionStore(),
            event_store=SubagentEventStore(),
            runtimes=args.get("runtimes"),
            cases=args.get("cases"),
            mode=args.get("mode"),
            live=bool(args.get("live", False)),
            project_id=args.get("project_id") or "OrynWorkspace",
            max_cases=args.get("max_cases") or 3,
            iterations=args.get("iterations") or 1,
            timeout_seconds=args.get("timeout_seconds") or 180,
            persist=bool(args.get("persist", True)),
        )
    except Exception as exc:
        return tool_error(f"Dev harness benchmark failed: {exc}")
    return tool_result(result)


def _handle_dev_harness_benchmark_runs(args: Dict[str, Any], **kwargs) -> str:
    try:
        store = DevExecutionStore()
        benchmark_run_id = str(args.get("benchmark_run_id") or "").strip()
        if benchmark_run_id:
            result = get_harness_benchmark_run(store=store, benchmark_run_id=benchmark_run_id)
        else:
            result = list_harness_benchmark_runs(store=store, limit=args.get("limit") or 50)
    except KeyError as exc:
        return tool_error(str(exc))
    except Exception as exc:
        return tool_error(f"Dev harness benchmark lookup failed: {exc}")
    return tool_result(result)


def _handle_dev_start_clarification(args: Dict[str, Any], **kwargs) -> str:
    try:
        result = start_clarification(
            store=DevClarificationStore(),
            vision_brief=args.get("vision_brief") or "",
            project_id=args.get("project_id") or "OrynWorkspace",
            session_id=args.get("session_id"),
            project_context=args.get("project_context"),
            max_questions=args.get("max_questions") or 5,
        )
    except Exception as exc:
        return tool_error(f"Dev clarification start failed: {exc}")
    return tool_result(result)


def _handle_dev_clarifications(args: Dict[str, Any], **kwargs) -> str:
    try:
        store = DevClarificationStore()
        clarification_id = str(args.get("clarification_id") or "").strip()
        if clarification_id:
            result = get_clarification(store=store, clarification_id=clarification_id)
        else:
            result = list_clarifications(
                store=store,
                project_id=args.get("project_id"),
                session_id=args.get("session_id"),
                status=args.get("status"),
                limit=args.get("limit") or 50,
            )
    except KeyError as exc:
        return tool_error(str(exc))
    except Exception as exc:
        return tool_error(f"Dev clarification lookup failed: {exc}")
    return tool_result(result)


def _handle_dev_answer_clarification(args: Dict[str, Any], **kwargs) -> str:
    try:
        result = answer_clarification(
            store=DevClarificationStore(),
            clarification_id=args.get("clarification_id") or "",
            question_id=args.get("question_id"),
            option_id=args.get("option_id"),
            answer_text=args.get("answer_text"),
            skipped=bool(args.get("skipped", False)),
            back=bool(args.get("back", False)),
        )
    except Exception as exc:
        return tool_error(f"Dev clarification answer failed: {exc}")
    return tool_result(result)


def _handle_dev_complete_clarification(args: Dict[str, Any], **kwargs) -> str:
    try:
        result = complete_clarification(
            store=DevClarificationStore(),
            clarification_id=args.get("clarification_id") or "",
        )
    except Exception as exc:
        return tool_error(f"Dev clarification complete failed: {exc}")
    return tool_result(result)


def _handle_dev_cancel_clarification(args: Dict[str, Any], **kwargs) -> str:
    try:
        result = cancel_clarification(
            store=DevClarificationStore(),
            clarification_id=args.get("clarification_id") or "",
            reason=args.get("reason"),
        )
    except Exception as exc:
        return tool_error(f"Dev clarification cancel failed: {exc}")
    return tool_result(result)


def _plan_artifact_stores() -> tuple[DevPlanArtifactStore, DevClarificationStore]:
    execution_store = DevExecutionStore()
    return (
        DevPlanArtifactStore(db_path=execution_store.db_path),
        DevClarificationStore(db_path=execution_store.db_path),
    )


def _handle_dev_plan_artifacts(args: Dict[str, Any], **kwargs) -> str:
    try:
        store, _ = _plan_artifact_stores()
        plan_artifact_id = str(args.get("plan_artifact_id") or "").strip()
        if plan_artifact_id:
            result = get_plan_artifact(store=store, plan_artifact_id=plan_artifact_id)
        else:
            result = list_plan_artifacts(
                store=store,
                clarification_id=args.get("clarification_id"),
                project_id=args.get("project_id"),
                status=args.get("status"),
                limit=args.get("limit") or 50,
            )
    except KeyError as exc:
        return tool_error(str(exc))
    except Exception as exc:
        return tool_error(f"Dev plan artifact lookup failed: {exc}")
    return tool_result(result)


def _handle_dev_create_plan_artifact(args: Dict[str, Any], **kwargs) -> str:
    try:
        store, clarification_store = _plan_artifact_stores()
        result = create_plan_artifact(
            store=store,
            clarification_store=clarification_store,
            clarification_id=args.get("clarification_id") or "",
        )
    except Exception as exc:
        return tool_error(f"Dev plan artifact creation failed: {exc}")
    return tool_result(result)


def _handle_dev_revise_plan_artifact(args: Dict[str, Any], **kwargs) -> str:
    try:
        store, clarification_store = _plan_artifact_stores()
        result = revise_plan_artifact(
            store=store,
            clarification_store=clarification_store,
            plan_artifact_id=args.get("plan_artifact_id") or "",
            feedback=args.get("feedback_instruction") or args.get("feedback") or "",
        )
    except Exception as exc:
        return tool_error(f"Dev plan artifact revision failed: {exc}")
    return tool_result(result)


def _handle_dev_approve_plan_artifact(args: Dict[str, Any], **kwargs) -> str:
    try:
        store, _ = _plan_artifact_stores()
        result = approve_plan_artifact(
            store=store,
            plan_artifact_id=args.get("plan_artifact_id") or "",
        )
    except Exception as exc:
        return tool_error(f"Dev plan artifact approval failed: {exc}")
    return tool_result(result)


def _handle_dev_cancel_plan_artifact(args: Dict[str, Any], **kwargs) -> str:
    try:
        store, _ = _plan_artifact_stores()
        result = cancel_plan_artifact(
            store=store,
            plan_artifact_id=args.get("plan_artifact_id") or "",
            reason=args.get("reason"),
        )
    except Exception as exc:
        return tool_error(f"Dev plan artifact cancel failed: {exc}")
    return tool_result(result)


def _handle_dev_create_execution_plan_from_artifact(args: Dict[str, Any], **kwargs) -> str:
    try:
        artifact_store, _ = _plan_artifact_stores()
        result = create_execution_plan_from_artifact(
            artifact_store=artifact_store,
            execution_store=DevExecutionStore(),
            plan_artifact_id=args.get("plan_artifact_id") or "",
        )
    except Exception as exc:
        return tool_error(f"Dev artifact build failed: {exc}")
    return tool_result(result)


def _handle_dev_plan_artifact_builds(args: Dict[str, Any], **kwargs) -> str:
    try:
        store, _ = _plan_artifact_stores()
        result = list_plan_artifact_builds(
            store=store,
            plan_artifact_id=args.get("plan_artifact_id") or "",
            limit=args.get("limit") or 25,
        )
    except Exception as exc:
        return tool_error(f"Dev artifact build lookup failed: {exc}")
    return tool_result(result)


def _handle_dev_execution_plan_draft_review(args: Dict[str, Any], **kwargs) -> str:
    try:
        result = get_execution_plan_draft_review(
            execution_store=DevExecutionStore(),
            plan_id=args.get("plan_id") or "",
        )
    except Exception as exc:
        return tool_error(f"Dev draft review lookup failed: {exc}")
    return tool_result(result)


def _handle_dev_revise_execution_plan_draft(args: Dict[str, Any], **kwargs) -> str:
    try:
        artifact_store, _ = _plan_artifact_stores()
        result = revise_execution_plan_draft(
            artifact_store=artifact_store,
            execution_store=DevExecutionStore(),
            plan_id=args.get("plan_id") or "",
            feedback=args.get("feedback_instruction") or args.get("feedback") or "",
        )
    except Exception as exc:
        return tool_error(f"Dev draft revision failed: {exc}")
    return tool_result(result)


def _handle_dev_approve_execution_plan_draft(args: Dict[str, Any], **kwargs) -> str:
    try:
        result = approve_execution_plan_draft(
            execution_store=DevExecutionStore(),
            plan_id=args.get("plan_id") or "",
        )
    except Exception as exc:
        return tool_error(f"Dev draft approval failed: {exc}")
    return tool_result(result)


def _handle_dev_cancel_execution_plan_draft(args: Dict[str, Any], **kwargs) -> str:
    try:
        result = cancel_execution_plan_draft(
            execution_store=DevExecutionStore(),
            plan_id=args.get("plan_id") or "",
            reason=args.get("reason"),
        )
    except Exception as exc:
        return tool_error(f"Dev draft cancellation failed: {exc}")
    return tool_result(result)


def _handle_dev_openhands_server_status(args: Dict[str, Any], **kwargs) -> str:
    return tool_result(openhands_server_status())


def _handle_dev_start_openhands_server(args: Dict[str, Any], **kwargs) -> str:
    try:
        result = start_openhands_server(
            cwd=args.get("cwd"),
            server_url=args.get("server_url"),
            wait_seconds=args.get("wait_seconds") or 5.0,
        )
    except Exception as exc:
        return tool_error(str(exc))
    return tool_result(result)


def _handle_dev_stop_openhands_server(args: Dict[str, Any], **kwargs) -> str:
    try:
        result = stop_openhands_server()
    except Exception as exc:
        return tool_error(str(exc))
    return tool_result(result)


def _handle_dev_create_execution_plan(args: Dict[str, Any], **kwargs) -> str:
    try:
        store = DevExecutionStore()
        plan = store.create_plan(
            title=args.get("title") or "Dev execution plan",
            vision_brief=args.get("vision_brief"),
            tasks=args.get("tasks") or [],
            runbook_id=args.get("runbook_id"),
            policy_profile=args.get("policy_profile"),
        )
    except Exception as exc:
        return tool_error(str(exc))
    return tool_result({"ok": True, "plan": plan})


def _handle_dev_launch_execution_plan(args: Dict[str, Any], **kwargs) -> str:
    plan_id = str(args.get("plan_id") or "").strip()
    if not plan_id:
        return tool_error("dev_launch_execution_plan requires plan_id")
    try:
        store = DevExecutionStore()
        result = launch_execution_plan(
            store=store,
            plan_id=plan_id,
            task_ids=args.get("task_ids") or None,
        )
    except KeyError as exc:
        return tool_error(str(exc))
    except Exception as exc:
        return tool_error(f"Dev plan launch failed: {exc}")
    return tool_result(result)


def _handle_dev_execution_plan_status(args: Dict[str, Any], **kwargs) -> str:
    plan_id = str(args.get("plan_id") or "").strip()
    if not plan_id:
        return tool_error("dev_execution_plan_status requires plan_id")
    try:
        result = derive_execution_plan_status(
            store=DevExecutionStore(),
            plan_id=plan_id,
            event_store=SubagentEventStore(),
        )
    except KeyError as exc:
        return tool_error(str(exc))
    except Exception as exc:
        return tool_error(f"Dev plan status failed: {exc}")
    return tool_result(result)


def _handle_dev_synthesize_execution_plan(args: Dict[str, Any], **kwargs) -> str:
    plan_id = str(args.get("plan_id") or "").strip()
    if not plan_id:
        return tool_error("dev_synthesize_execution_plan requires plan_id")
    try:
        result = synthesize_execution_plan(
            store=DevExecutionStore(),
            plan_id=plan_id,
            event_store=SubagentEventStore(),
        )
    except KeyError as exc:
        return tool_error(str(exc))
    except Exception as exc:
        return tool_error(f"Dev plan synthesis failed: {exc}")
    return tool_result(result)


def _handle_dev_review_execution_plan(args: Dict[str, Any], **kwargs) -> str:
    plan_id = str(args.get("plan_id") or "").strip()
    if not plan_id:
        return tool_error("dev_review_execution_plan requires plan_id")
    try:
        result = review_execution_plan(
            store=DevExecutionStore(),
            plan_id=plan_id,
            event_store=SubagentEventStore(),
            include_synthesis=bool(args.get("include_synthesis", True)),
        )
    except KeyError as exc:
        return tool_error(str(exc))
    except Exception as exc:
        return tool_error(f"Dev plan review failed: {exc}")
    return tool_result(result)


def _handle_dev_apply_execution_plan_review(args: Dict[str, Any], **kwargs) -> str:
    plan_id = str(args.get("plan_id") or "").strip()
    if not plan_id:
        return tool_error("dev_apply_execution_plan_review requires plan_id")
    try:
        result = apply_execution_plan_review(
            store=DevExecutionStore(),
            plan_id=plan_id,
            event_store=SubagentEventStore(),
            include_synthesis=bool(args.get("include_synthesis", True)),
            message=args.get("message"),
            instruction=args.get("instruction"),
            project_id=args.get("project_id"),
            agent=args.get("agent"),
            model=args.get("model"),
            reasoning_effort=args.get("reasoning_effort"),
        )
    except KeyError as exc:
        return tool_error(str(exc))
    except Exception as exc:
        return tool_error(f"Dev plan review application failed: {exc}")
    return tool_result(result)


def _handle_dev_supervise_execution_plans(args: Dict[str, Any], **kwargs) -> str:
    try:
        result = supervise_execution_plans(
            store=DevExecutionStore(),
            plan_ids=args.get("plan_ids") or None,
            limit=int(args.get("limit") or 20),
            project_id=args.get("project_id") or None,
            reviewable_only=bool(args.get("reviewable_only", False)),
            event_store=SubagentEventStore(),
            apply_guarded_actions=bool(args.get("apply_guarded_actions", True)),
            include_synthesis=bool(args.get("include_synthesis", False)),
        )
    except Exception as exc:
        return tool_error(f"Dev plan supervision failed: {exc}")
    return tool_result(result)


def _handle_dev_runbooks(args: Dict[str, Any], **kwargs) -> str:
    runbook_id = str(args.get("runbook_id") or "").strip()
    try:
        if runbook_id:
            result = get_runbook(store=DevExecutionStore(), runbook_id=runbook_id)
        else:
            result = list_runbooks(
                store=DevExecutionStore(),
                project_id=args.get("project_id") or None,
                limit=int(args.get("limit") or 100),
            )
    except KeyError as exc:
        return tool_error(str(exc))
    except Exception as exc:
        return tool_error(f"Dev runbooks failed: {exc}")
    return tool_result(result)


def _handle_dev_set_project_runbook(args: Dict[str, Any], **kwargs) -> str:
    project_id = str(args.get("project_id") or "").strip()
    if not project_id:
        return tool_error("dev_set_project_runbook requires project_id")
    try:
        result = set_project_runbook(
            store=DevExecutionStore(),
            project_id=project_id,
            policy_profile=args.get("policy_profile") or "standard",
            max_follow_ups_per_task=args.get("max_follow_ups_per_task"),
            max_retries_per_task=args.get("max_retries_per_task"),
            supervisor_enabled=args.get("supervisor_enabled"),
            supervisor_interval_seconds=args.get("supervisor_interval_seconds"),
            supervisor_limit=args.get("supervisor_limit"),
            supervisor_include_synthesis=args.get("supervisor_include_synthesis"),
            supervisor_apply_guarded_actions=args.get("supervisor_apply_guarded_actions"),
        )
    except Exception as exc:
        return tool_error(f"Dev set project runbook failed: {exc}")
    return tool_result(result)


def _handle_dev_supervisor_loop_status(args: Dict[str, Any], **kwargs) -> str:
    try:
        result = list_supervisor_loop_status(
            store=DevExecutionStore(),
            project_id=args.get("project_id") or None,
        )
    except Exception as exc:
        return tool_error(f"Dev supervisor loop status failed: {exc}")
    return tool_result(result)


def _handle_dev_set_supervisor_loop(args: Dict[str, Any], **kwargs) -> str:
    project_id = str(args.get("project_id") or "").strip()
    if not project_id:
        return tool_error("dev_set_supervisor_loop requires project_id")
    try:
        result = set_supervisor_loop(
            store=DevExecutionStore(),
            project_id=project_id,
            supervisor_enabled=args.get("supervisor_enabled"),
            supervisor_interval_seconds=args.get("supervisor_interval_seconds"),
            supervisor_limit=args.get("supervisor_limit"),
            supervisor_include_synthesis=args.get("supervisor_include_synthesis"),
            supervisor_apply_guarded_actions=args.get("supervisor_apply_guarded_actions"),
        )
    except Exception as exc:
        return tool_error(f"Dev set supervisor loop failed: {exc}")
    return tool_result(result)


def _handle_dev_next_execution_step(args: Dict[str, Any], **kwargs) -> str:
    plan_id = str(args.get("plan_id") or "").strip()
    if not plan_id:
        return tool_error("dev_next_execution_step requires plan_id")
    try:
        result = next_execution_step(
            store=DevExecutionStore(),
            plan_id=plan_id,
            event_store=SubagentEventStore(),
            include_synthesis=bool(args.get("include_synthesis", False)),
        )
    except KeyError as exc:
        return tool_error(str(exc))
    except Exception as exc:
        return tool_error(f"Dev next execution step failed: {exc}")
    return tool_result(result)


def _handle_dev_set_execution_plan_test_state(args: Dict[str, Any], **kwargs) -> str:
    plan_id = str(args.get("plan_id") or "").strip()
    task_id = str(args.get("task_id") or "").strip()
    state = str(args.get("state") or "").strip()
    if not plan_id or not task_id or not state:
        return tool_error("dev_set_execution_plan_test_state requires plan_id, task_id, and state")
    try:
        result = set_execution_plan_test_state(
            store=DevExecutionStore(),
            plan_id=plan_id,
            task_id=task_id,
            state=state,
            event_store=SubagentEventStore(),
            summary=args.get("summary"),
            status_reason=args.get("status_reason"),
            ao_session_id=args.get("ao_session_id"),
            project_id=args.get("project_id"),
            files_read=args.get("files_read") if isinstance(args.get("files_read"), list) else None,
            files_written=args.get("files_written") if isinstance(args.get("files_written"), list) else None,
            verification_evidence=args.get("verification_evidence") if isinstance(args.get("verification_evidence"), list) else None,
        )
    except KeyError as exc:
        return tool_error(str(exc))
    except Exception as exc:
        return tool_error(f"Dev test state injection failed: {exc}")
    return tool_result(result)


def _handle_dev_supervisor_approvals(args: Dict[str, Any], **kwargs) -> str:
    approval_id = str(args.get("approval_id") or "").strip()
    try:
        if approval_id:
            result = get_supervisor_approval(
                store=DevExecutionStore(),
                approval_id=approval_id,
            )
        else:
            result = list_supervisor_approvals(
                store=DevExecutionStore(),
                status=args.get("status") or None,
                plan_id=args.get("plan_id") or None,
                limit=int(args.get("limit") or 50),
            )
    except KeyError as exc:
        return tool_error(str(exc))
    except Exception as exc:
        return tool_error(f"Dev supervisor approvals failed: {exc}")
    return tool_result(result)


def _handle_dev_approve_supervisor_action(args: Dict[str, Any], **kwargs) -> str:
    approval_id = str(args.get("approval_id") or "").strip()
    if not approval_id:
        return tool_error("dev_approve_supervisor_action requires approval_id")
    try:
        result = approve_supervisor_approval(
            store=DevExecutionStore(),
            approval_id=approval_id,
            message=args.get("message"),
            instruction=args.get("instruction"),
            project_id=args.get("project_id"),
            agent=args.get("agent"),
            model=args.get("model"),
            reasoning_effort=args.get("reasoning_effort"),
        )
    except (KeyError, ValueError) as exc:
        return tool_error(str(exc))
    except Exception as exc:
        return tool_error(f"Dev supervisor approve failed: {exc}")
    return tool_result(result)


def _handle_dev_deny_supervisor_action(args: Dict[str, Any], **kwargs) -> str:
    approval_id = str(args.get("approval_id") or "").strip()
    if not approval_id:
        return tool_error("dev_deny_supervisor_action requires approval_id")
    try:
        result = deny_supervisor_approval(
            store=DevExecutionStore(),
            approval_id=approval_id,
            message=args.get("message"),
        )
    except (KeyError, ValueError) as exc:
        return tool_error(str(exc))
    except Exception as exc:
        return tool_error(f"Dev supervisor deny failed: {exc}")
    return tool_result(result)


def _handle_dev_apply_supervisor_approval(args: Dict[str, Any], **kwargs) -> str:
    approval_id = str(args.get("approval_id") or "").strip()
    if not approval_id:
        return tool_error("dev_apply_supervisor_approval requires approval_id")
    try:
        result = apply_supervisor_approval(
            store=DevExecutionStore(),
            approval_id=approval_id,
            event_store=SubagentEventStore(),
            include_synthesis=bool(args.get("include_synthesis", True)),
        )
    except KeyError as exc:
        return tool_error(str(exc))
    except Exception as exc:
        return tool_error(f"Dev supervisor apply failed: {exc}")
    return tool_result(result)


registry.register(
    name="dev_launch_profiles",
    toolset="delegation",
    schema=DEV_LAUNCH_PROFILES_SCHEMA,
    handler=_handle_dev_launch_profiles,
    emoji="Dev",
    max_result_size_chars=20_000,
)

registry.register(
    name="dev_worker_runtimes",
    toolset="delegation",
    schema=DEV_WORKER_RUNTIMES_SCHEMA,
    handler=_handle_dev_worker_runtimes,
    emoji="Dev",
    max_result_size_chars=20_000,
)

registry.register(
    name="dev_select_worker_runtime",
    toolset="delegation",
    schema=DEV_SELECT_WORKER_RUNTIME_SCHEMA,
    handler=_handle_dev_select_worker_runtime,
    emoji="Dev",
    max_result_size_chars=20_000,
)

registry.register(
    name="dev_harness_components",
    toolset="delegation",
    schema=DEV_HARNESS_COMPONENTS_SCHEMA,
    handler=_handle_dev_harness_components,
    emoji="Dev",
    max_result_size_chars=30_000,
)

registry.register(
    name="dev_harness_report",
    toolset="delegation",
    schema=DEV_HARNESS_REPORT_SCHEMA,
    handler=_handle_dev_harness_report,
    emoji="Dev",
    max_result_size_chars=60_000,
)

registry.register(
    name="dev_harness_recommendations",
    toolset="delegation",
    schema=DEV_HARNESS_RECOMMENDATIONS_SCHEMA,
    handler=_handle_dev_harness_recommendations,
    emoji="Dev",
    max_result_size_chars=60_000,
)

registry.register(
    name="dev_harness_recommendation_runs",
    toolset="delegation",
    schema=DEV_HARNESS_RECOMMENDATION_RUNS_SCHEMA,
    handler=_handle_dev_harness_recommendation_runs,
    emoji="Dev",
    max_result_size_chars=60_000,
)

registry.register(
    name="dev_harness_benchmark",
    toolset="delegation",
    schema=DEV_HARNESS_BENCHMARK_SCHEMA,
    handler=_handle_dev_harness_benchmark,
    emoji="Dev",
    max_result_size_chars=60_000,
)

registry.register(
    name="dev_harness_benchmark_runs",
    toolset="delegation",
    schema=DEV_HARNESS_BENCHMARK_RUNS_SCHEMA,
    handler=_handle_dev_harness_benchmark_runs,
    emoji="Dev",
    max_result_size_chars=60_000,
)

registry.register(
    name="dev_start_clarification",
    toolset="delegation",
    schema=DEV_START_CLARIFICATION_SCHEMA,
    handler=_handle_dev_start_clarification,
    emoji="Dev",
    max_result_size_chars=40_000,
)

registry.register(
    name="dev_clarifications",
    toolset="delegation",
    schema=DEV_CLARIFICATIONS_SCHEMA,
    handler=_handle_dev_clarifications,
    emoji="Dev",
    max_result_size_chars=40_000,
)

registry.register(
    name="dev_answer_clarification",
    toolset="delegation",
    schema=DEV_ANSWER_CLARIFICATION_SCHEMA,
    handler=_handle_dev_answer_clarification,
    emoji="Dev",
    max_result_size_chars=40_000,
)

registry.register(
    name="dev_complete_clarification",
    toolset="delegation",
    schema=DEV_COMPLETE_CLARIFICATION_SCHEMA,
    handler=_handle_dev_complete_clarification,
    emoji="Dev",
    max_result_size_chars=40_000,
)

registry.register(
    name="dev_cancel_clarification",
    toolset="delegation",
    schema=DEV_CANCEL_CLARIFICATION_SCHEMA,
    handler=_handle_dev_cancel_clarification,
    emoji="Dev",
    max_result_size_chars=40_000,
)

registry.register(
    name="dev_plan_artifacts",
    toolset="delegation",
    schema=DEV_PLAN_ARTIFACTS_SCHEMA,
    handler=_handle_dev_plan_artifacts,
    emoji="Dev",
    max_result_size_chars=60_000,
)

registry.register(
    name="dev_create_plan_artifact",
    toolset="delegation",
    schema=DEV_CREATE_PLAN_ARTIFACT_SCHEMA,
    handler=_handle_dev_create_plan_artifact,
    emoji="Dev",
    max_result_size_chars=60_000,
)

registry.register(
    name="dev_revise_plan_artifact",
    toolset="delegation",
    schema=DEV_REVISE_PLAN_ARTIFACT_SCHEMA,
    handler=_handle_dev_revise_plan_artifact,
    emoji="Dev",
    max_result_size_chars=60_000,
)

registry.register(
    name="dev_approve_plan_artifact",
    toolset="delegation",
    schema=DEV_APPROVE_PLAN_ARTIFACT_SCHEMA,
    handler=_handle_dev_approve_plan_artifact,
    emoji="Dev",
    max_result_size_chars=60_000,
)

registry.register(
    name="dev_cancel_plan_artifact",
    toolset="delegation",
    schema=DEV_CANCEL_PLAN_ARTIFACT_SCHEMA,
    handler=_handle_dev_cancel_plan_artifact,
    emoji="Dev",
    max_result_size_chars=60_000,
)

registry.register(
    name="dev_create_execution_plan_from_artifact",
    toolset="delegation",
    schema=DEV_CREATE_EXECUTION_PLAN_FROM_ARTIFACT_SCHEMA,
    handler=_handle_dev_create_execution_plan_from_artifact,
    emoji="Dev",
    max_result_size_chars=80_000,
)

registry.register(
    name="dev_plan_artifact_builds",
    toolset="delegation",
    schema=DEV_PLAN_ARTIFACT_BUILDS_SCHEMA,
    handler=_handle_dev_plan_artifact_builds,
    emoji="Dev",
    max_result_size_chars=60_000,
)

registry.register(
    name="dev_execution_plan_draft_review",
    toolset="delegation",
    schema=DEV_EXECUTION_PLAN_DRAFT_REVIEW_SCHEMA,
    handler=_handle_dev_execution_plan_draft_review,
    emoji="Dev",
    max_result_size_chars=80_000,
)

registry.register(
    name="dev_revise_execution_plan_draft",
    toolset="delegation",
    schema=DEV_REVISE_EXECUTION_PLAN_DRAFT_SCHEMA,
    handler=_handle_dev_revise_execution_plan_draft,
    emoji="Dev",
    max_result_size_chars=100_000,
)

registry.register(
    name="dev_approve_execution_plan_draft",
    toolset="delegation",
    schema=DEV_APPROVE_EXECUTION_PLAN_DRAFT_SCHEMA,
    handler=_handle_dev_approve_execution_plan_draft,
    emoji="Dev",
    max_result_size_chars=80_000,
)

registry.register(
    name="dev_cancel_execution_plan_draft",
    toolset="delegation",
    schema=DEV_CANCEL_EXECUTION_PLAN_DRAFT_SCHEMA,
    handler=_handle_dev_cancel_execution_plan_draft,
    emoji="Dev",
    max_result_size_chars=80_000,
)

registry.register(
    name="dev_openhands_server_status",
    toolset="delegation",
    schema=DEV_OPENHANDS_SERVER_STATUS_SCHEMA,
    handler=_handle_dev_openhands_server_status,
    emoji="Dev",
    max_result_size_chars=20_000,
)

registry.register(
    name="dev_start_openhands_server",
    toolset="delegation",
    schema=DEV_START_OPENHANDS_SERVER_SCHEMA,
    handler=_handle_dev_start_openhands_server,
    emoji="Dev",
    max_result_size_chars=20_000,
)

registry.register(
    name="dev_stop_openhands_server",
    toolset="delegation",
    schema=DEV_STOP_OPENHANDS_SERVER_SCHEMA,
    handler=_handle_dev_stop_openhands_server,
    emoji="Dev",
    max_result_size_chars=20_000,
)

registry.register(
    name="dev_create_execution_plan",
    toolset="delegation",
    schema=DEV_CREATE_EXECUTION_PLAN_SCHEMA,
    handler=_handle_dev_create_execution_plan,
    emoji="Dev",
    max_result_size_chars=30_000,
)

registry.register(
    name="dev_launch_execution_plan",
    toolset="delegation",
    schema=DEV_LAUNCH_EXECUTION_PLAN_SCHEMA,
    handler=_handle_dev_launch_execution_plan,
    emoji="Dev",
    max_result_size_chars=30_000,
)

registry.register(
    name="dev_execution_plan_status",
    toolset="delegation",
    schema=DEV_EXECUTION_PLAN_STATUS_SCHEMA,
    handler=_handle_dev_execution_plan_status,
    emoji="Dev",
    max_result_size_chars=30_000,
)

registry.register(
    name="dev_synthesize_execution_plan",
    toolset="delegation",
    schema=DEV_SYNTHESIZE_EXECUTION_PLAN_SCHEMA,
    handler=_handle_dev_synthesize_execution_plan,
    emoji="Dev",
    max_result_size_chars=40_000,
)

registry.register(
    name="dev_review_execution_plan",
    toolset="delegation",
    schema=DEV_REVIEW_EXECUTION_PLAN_SCHEMA,
    handler=_handle_dev_review_execution_plan,
    emoji="Dev",
    max_result_size_chars=40_000,
)

registry.register(
    name="dev_apply_execution_plan_review",
    toolset="delegation",
    schema=DEV_APPLY_EXECUTION_PLAN_REVIEW_SCHEMA,
    handler=_handle_dev_apply_execution_plan_review,
    emoji="Dev",
    max_result_size_chars=50_000,
)

registry.register(
    name="dev_supervise_execution_plans",
    toolset="delegation",
    schema=DEV_SUPERVISE_EXECUTION_PLANS_SCHEMA,
    handler=_handle_dev_supervise_execution_plans,
    emoji="Dev",
    max_result_size_chars=50_000,
)

registry.register(
    name="dev_runbooks",
    toolset="delegation",
    schema=DEV_RUNBOOKS_SCHEMA,
    handler=_handle_dev_runbooks,
    emoji="Dev",
    max_result_size_chars=30_000,
)

registry.register(
    name="dev_set_project_runbook",
    toolset="delegation",
    schema=DEV_SET_PROJECT_RUNBOOK_SCHEMA,
    handler=_handle_dev_set_project_runbook,
    emoji="Dev",
    max_result_size_chars=30_000,
)

registry.register(
    name="dev_supervisor_loop_status",
    toolset="delegation",
    schema=DEV_SUPERVISOR_LOOP_STATUS_SCHEMA,
    handler=_handle_dev_supervisor_loop_status,
    emoji="Dev",
    max_result_size_chars=30_000,
)

registry.register(
    name="dev_set_supervisor_loop",
    toolset="delegation",
    schema=DEV_SET_SUPERVISOR_LOOP_SCHEMA,
    handler=_handle_dev_set_supervisor_loop,
    emoji="Dev",
    max_result_size_chars=30_000,
)

registry.register(
    name="dev_next_execution_step",
    toolset="delegation",
    schema=DEV_NEXT_EXECUTION_STEP_SCHEMA,
    handler=_handle_dev_next_execution_step,
    emoji="Dev",
    max_result_size_chars=50_000,
)

registry.register(
    name="dev_set_execution_plan_test_state",
    toolset="delegation",
    schema=DEV_SET_EXECUTION_PLAN_TEST_STATE_SCHEMA,
    handler=_handle_dev_set_execution_plan_test_state,
    emoji="Dev",
    max_result_size_chars=50_000,
)

registry.register(
    name="dev_supervisor_approvals",
    toolset="delegation",
    schema=DEV_SUPERVISOR_APPROVALS_SCHEMA,
    handler=_handle_dev_supervisor_approvals,
    emoji="Dev",
    max_result_size_chars=50_000,
)

registry.register(
    name="dev_approve_supervisor_action",
    toolset="delegation",
    schema=DEV_APPROVE_SUPERVISOR_ACTION_SCHEMA,
    handler=_handle_dev_approve_supervisor_action,
    emoji="Dev",
    max_result_size_chars=30_000,
)

registry.register(
    name="dev_deny_supervisor_action",
    toolset="delegation",
    schema=DEV_DENY_SUPERVISOR_ACTION_SCHEMA,
    handler=_handle_dev_deny_supervisor_action,
    emoji="Dev",
    max_result_size_chars=30_000,
)

registry.register(
    name="dev_apply_supervisor_approval",
    toolset="delegation",
    schema=DEV_APPLY_SUPERVISOR_APPROVAL_SCHEMA,
    handler=_handle_dev_apply_supervisor_approval,
    emoji="Dev",
    max_result_size_chars=50_000,
)
