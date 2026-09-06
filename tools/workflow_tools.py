#!/usr/bin/env python3
"""Read and edit the workflow open on the Workflows canvas in the Hermes desktop GUI.

The canvas is a node graph — steps an agent runs, the wires between them, and
the gates that branch on what happened. It lives entirely in the desktop
renderer (the ``workflows`` GUI plugin owns the document and its storage), so
this tool round-trips through the gateway's blocking-prompt bridge the same way
``read_terminal`` and ``drive_preview`` do: tui_gateway emits
``workflow.request``, the plugin applies the change to the live canvas and
answers ``workflow.respond``.

The point of the arrangement is that there is no privileged path. An edit from
here runs through the SAME dispatcher the inspector and the drag handles run
through, so an agent edit and a hand edit are the same operation on the same
document and land in the same undo history. The user can watch it happen.

Deliberately ONE tool with an ``action`` rather than a dozen ``graph_*`` tools:
the op vocabulary is defined in the plugin (TypeScript), it is the thing most
likely to grow, and mirroring it into a Python schema would guarantee the two
drift. Instead ``action="read"`` returns the live contract along with the
graph, which is why the schema below tells the model to call it first.

Lives in the ``desktop_ui`` toolset, which the GUI gateway enables only for
desktop-sourced sessions.
"""

import json
from typing import Callable, Optional

from tools.registry import registry, tool_error

ACTIONS = ("read", "edit", "list", "open", "create", "run")

# Verbs that name a workflow to act on. `create` takes a name for a new one;
# `open` / `run` take an id or name of an existing one.
NEEDS_WORKFLOW = ("open", "create", "run")


def workflow_tool(
    action: str = "",
    ops: Optional[list] = None,
    workflow: Optional[str] = None,
    scenario: Optional[dict] = None,
    payload: Optional[object] = None,
    callback: Optional[Callable] = None,
) -> str:
    """Dispatch one canvas action, or start a stored run on the gateway."""
    verb = (action or "").strip().lower()
    if verb not in ACTIONS:
        return tool_error(f"action must be one of: {', '.join(ACTIONS)}.")

    if verb in NEEDS_WORKFLOW and not (workflow or "").strip():
        return tool_error(f"{verb} needs a workflow name.")

    # A run is the gateway walking the stored graph. It does not need the
    # canvas — that is the point of the HERMES_HOME copy.
    if verb == "run":
        try:
            from workflow.runner import start_run

            state = start_run(str(workflow).strip(), payload=payload, source="tool")
        except ValueError as exc:
            return tool_error(str(exc))
        except Exception as exc:
            return tool_error(f"Failed to start the run: {exc}")
        return json.dumps(
            {
                "runId": state.get("runId"),
                "status": state.get("status"),
                "workflow": state.get("workflowId"),
            },
            ensure_ascii=False,
        )

    if callback is None:
        return tool_error("workflow is only available in the Hermes desktop app.")

    if verb == "edit":
        if not isinstance(ops, list) or not ops:
            return tool_error(
                "edit needs a non-empty ops array, e.g. "
                '[{"tool": "graph_add_step", "args": {"kind": "agent", "title": "Lint"}}]. '
                "Call action='read' first for the available ops."
            )
        bad = [op for op in ops if not isinstance(op, dict) or not op.get("tool")]
        if bad:
            return tool_error('every op needs a "tool" name, e.g. {"tool": "graph_connect", "args": {...}}.')

    payload = {
        name: val
        for name, val in (
            ("action", verb),
            ("ops", ops),
            ("workflow", workflow),
            ("scenario", scenario),
        )
        if val is not None
    }

    try:
        raw = callback(payload)
    except Exception as exc:
        return tool_error(f"Failed to reach the Workflows canvas: {exc}")

    if not raw:
        # The plugin answers whether or not its page is on screen, so silence
        # means it isn't loaded at all — which is the default, since Workflows
        # ships off. Name the switch rather than the page.
        return tool_error(
            "Nothing answered for the Workflows canvas. The Workflows plugin is off by "
            "default — the user can turn it on in Settings ▸ Plugins."
        )

    # The renderer answers with a JSON object; pass it through, else wrap it.
    try:
        return json.dumps(json.loads(raw), ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps({"text": str(raw)}, ensure_ascii=False)


WORKFLOW_SCHEMA = {
    "name": "workflow",
    "description": (
        "Read and edit the workflow open on the Workflows canvas in the Hermes "
        "desktop GUI — the node graph the user is looking at. A workflow is a "
        "graph of steps: 'trigger' (what starts a run), 'agent' (a model does "
        "the work), 'human' (a person does, and the run parks on them), "
        "'gate' (branches on what already happened), and 'wait' (holds for "
        "the world). "
        "ALWAYS call action='read' first. It returns the open workflow's "
        "scenario — every step with its id and config, every wire with its "
        "branch condition — plus the current validation problems AND the list "
        "of ops you may apply, each with its full JSON Schema. That op list is "
        "the authority on what 'edit' accepts; do not guess an op or an "
        "argument name from this description. "
        "action='edit' applies a batch of those ops in order, as one change: "
        "pass ops=[{tool, args}, ...]. Each op is checked and applied against "
        "the result of the one before, so you can add a step and wire it in the "
        "same call. The reply says what landed, what was refused and why, and "
        "the validation problems afterwards — read them, because a graph that "
        "doesn't validate won't run. "
        "Edits go through the same code path as the user's own drags and "
        "inspector edits, so they appear on the canvas immediately and are "
        "undoable with the rest of their work. Prefer the surgical ops for a "
        "change to an existing graph, and graph_set_scenario only when "
        "authoring a whole workflow at once. "
        "action='list' names every workflow the user has; action='open' "
        "switches the canvas to one of them (by id or name); action='create' "
        "makes a new one and opens it, optionally seeded with a scenario. Both "
        "open and create bring the canvas on screen. "
        "action='run' starts the stored workflow on the gateway (optional "
        "payload is handed to the first step). It does not need the canvas. "
        "To WALK someone through a workflow rather than describe it, pair this "
        "with the `tour` tool: every step card on the canvas is addressable as "
        "[data-tour=\"step:<id>\"] using the ids from action='read', so you can "
        "highlight each step in turn and narrate it on the user's screen. The "
        "canvas has to be open for that — action='open' first. "
        "Use this whenever the user asks you to build, inspect, fix, explain, "
        "or run one of their workflows."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(ACTIONS),
                "description": "What to do. Start with 'read'.",
            },
            "ops": {
                "type": "array",
                "description": (
                    "For 'edit': the ops to apply, in order. Each is "
                    '{"tool": "<op name>", "args": {...}} using an op and '
                    "argument shape from action='read'."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "tool": {
                            "type": "string",
                            "description": "Op name, e.g. 'graph_add_step'. From action='read'.",
                        },
                        "args": {
                            "type": "object",
                            "description": "That op's arguments, per its schema from action='read'.",
                        },
                    },
                    "required": ["tool"],
                },
            },
            "workflow": {
                "type": "string",
                "description": (
                    "For 'open' / 'run': the id or name of the workflow. "
                    "For 'create': the name of the new one."
                ),
            },
            "payload": {
                "description": "For 'run': optional trigger payload handed to the first step.",
            },
            "scenario": {
                "type": "object",
                "description": (
                    "For 'create': an optional starting scenario ({steps, edges}), "
                    "same shape as graph_set_scenario takes. Omit for an empty canvas."
                ),
            },
        },
        "required": ["action"],
    },
}


registry.register(
    name="workflow",
    toolset="desktop_ui",
    schema=WORKFLOW_SCHEMA,
    handler=lambda args, **kw: workflow_tool(
        action=args.get("action", ""),
        ops=args.get("ops"),
        workflow=args.get("workflow"),
        scenario=args.get("scenario"),
        payload=args.get("payload"),
        callback=kw.get("callback"),
    ),
    emoji="🕸️",
)
