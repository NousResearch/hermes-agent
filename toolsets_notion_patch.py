# =============================================================================
# NOTION TOOLSET — add this block to toolsets.py
# =============================================================================
#
# 1. In your imports section at the top of toolsets.py, add:
#
#    from tools.notion import NOTION_TOOLS
#
# 2. In the TOOLSETS dict, add the "notion" entry below.
#    (Place it alphabetically, e.g. between "moa" and "skills")
#
# =============================================================================

"notion": {
    "name": "notion",
    "description": (
        "Read and write Notion pages, databases, and blocks. "
        "Search for content, create pages, append notes, update properties, "
        "and query databases. Requires NOTION_API_KEY in ~/.hermes/.env."
    ),
    "tools": NOTION_TOOLS,
    "env_vars": ["NOTION_API_KEY"],
    "requires_packages": ["requests"],   # already a core dependency
},

# =============================================================================
# Also add "notion" to the TOOLSET_GROUPS dict if you use grouping, e.g.:
#
# "productivity": ["todo", "memory", "notion"],
# =============================================================================
