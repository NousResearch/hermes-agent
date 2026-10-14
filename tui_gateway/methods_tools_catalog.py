"""Command catalog data and stable presentation category identities."""

class Catalog:
    """Accumulator for commands.catalog: ``pairs`` (every [key, desc]), ``canon`` (lowercase
    key/alias → canonical key), ``commands`` (key → desktop meta) and ordered categories."""

    def __init__(self) -> None:
        self.pairs: list[list[str]] = []
        self.canon: dict[str, str] = {}
        self.commands: dict[str, dict[str, str | None]] = {}
        self.description_keys: dict[str, str] = {}
        self.cat_map: dict[str, list[list[str]]] = {}  # insertion order = category order

    def add(self, key: str, desc: str, cat: str, *, description_key: str | None = None) -> None:
        self.canon[key.lower()] = key
        self.pairs.append([key, desc])
        if description_key:
            self.description_keys[key] = description_key
        else:
            self.description_keys.pop(key, None)
        self.cat_map.setdefault(cat, []).append([key, desc])


def _command_category_key(category: str) -> str:
    """Stable presentation id; clients own localized category copy."""
    return {
        "Session": "session",
        "Configuration": "configuration",
        "Tools & Skills": "tools",
        "Info": "info",
        "Exit": "exit",
        "TUI": "tui",
        "User commands": "userCommands",
    }.get(category, "")
