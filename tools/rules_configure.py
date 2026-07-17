"""Tool registration for rules_configure.

The agent uses this tool to create, read, update, delete, and list
rules in the profile's ``rules/`` directory or in project-level
``.hermes/rules/`` directories.
"""

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
    return _run(
        action=args.get("action", ""),
        name=args.get("name"),
        body=args.get("body"),
        description=args.get("description"),
        always_apply=args.get("always_apply"),
        globs=args.get("globs"),
        overwrite=bool(args.get("overwrite", False)),
        scope=args.get("scope"),
    )


registry.register(
    name="rules_configure",
    toolset="config",
    schema=SCHEMA,
    handler=_handler,
    emoji="📐",
)
