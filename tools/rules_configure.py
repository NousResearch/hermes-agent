"""Tool registration for rules_configure.

The agent uses this tool to create, read, update, delete, and list
rules in the profile's ``rules/`` directory or in project-level
``.hermes/rules/`` directories.

Contract note: ``tools.registry._normalize_handler_result`` accepts
strings (and multimodal envelopes) only -- every other return shape is
coerced into a ``tool_result_contract`` error before reaching the
agent loop. The CLI (``hermes rules ...``) calls
``agent.rules_configure_tool.run`` directly and consumes the dict;
this *handler* stringifies that dict for the registry. Do not change
``run()`` to return a string -- the CLI depends on the structured
shape (#66441 review).
"""

import json

from tools.registry import registry
from agent.rules_configure_tool import run as _run


SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["create", "read", "update", "delete", "list"],
        },
        "name": {
            "type": "string",
            "description": (
                "Rule path under rules/, e.g. 'ui/conventions'. "
                "Omit for 'list'."
            ),
        },
        "body": {
            "type": "string",
            "description": "Markdown body (no frontmatter).",
        },
        "description": {
            "type": "string",
            "description": "Short label shown in the rule picker.",
        },
        "always_apply": {
            "type": "boolean",
            "description": (
                "Inject into system prompt on every session. "
                "Default true when globs is empty."
            ),
        },
        "globs": {
            "type": "array",
            "items": {"type": "string"},
            "description": "File patterns that activate this rule.",
        },
        "overwrite": {
            "type": "boolean",
            "description": "Required true to overwrite an existing rule.",
        },
        "scope": {
            "type": "string",
            "enum": ["profile", "project"],
            "description": (
                "'profile' -> ~/.hermes/profiles/<profile>/rules/ "
                "'project' -> nearest ./.hermes/rules/ (cwd-first). "
                "Default: 'project' if a .hermes/rules/ exists, else 'profile'."
            ),
        },
    },
    "required": ["action"],
}


def _handler(args, **_kwargs):
    result = _run(
        action=args.get("action", ""),
        name=args.get("name"),
        body=args.get("body"),
        description=args.get("description"),
        always_apply=args.get("always_apply"),
        globs=args.get("globs"),
        overwrite=bool(args.get("overwrite", False)),
        scope=args.get("scope"),
    )
    # Registry contract: tool handlers must return a string. ``run()``
    # returns a structured dict that the CLI consumes directly; here we
    # serialize it so the agent loop can read it. ``ensure_ascii=False``
    # so non-ASCII rule bodies survive round-trip.
    return json.dumps(result, ensure_ascii=False)


registry.register(
    name="rules_configure",
    toolset="config",
    schema=SCHEMA,
    handler=_handler,
    emoji="📐",
)
