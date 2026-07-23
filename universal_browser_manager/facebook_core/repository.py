"""Read-only repository for the canonical Facebook database."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .storage import canonical_db_path, connect


_FRIEND_COLUMNS = (
    "id, canonical_key, name, profile_url, thread_url, gender, status, "
    "is_friend, is_business, is_favorite, is_fake, permission_tier, "
    "profile_analysis, timeline_data, llm_analysis, relationship_tier, "
    "relationship_status, added_at, updated_at"
)


class AmbiguousFriendError(ValueError):
    """Raised when a name maps to more than one canonical CRM contact."""

    def __init__(self, friend_key: str, matching_ids: list[int]) -> None:
        self.friend_key = friend_key
        self.matching_ids = tuple(matching_ids)
        super().__init__(
            f"Ambiguous contact name; use CRM id. Matching ids: {matching_ids}"
        )


class FacebookRepository:
    """Read canonical Facebook records without mutating runtime state."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else canonical_db_path()

    def find_friend(self, friend_key: str | int) -> dict[str, Any] | None:
        """Resolve one friend by CRM id or one exact case-insensitive name."""
        key = str(friend_key).strip()
        if not key:
            return None

        connection = connect(self.db_path, readonly=True)
        try:
            if key.isdigit():
                row = connection.execute(
                    f"SELECT {_FRIEND_COLUMNS} FROM friends WHERE id = ?",
                    (int(key),),
                ).fetchone()
                if row is not None:
                    return dict(row)

            matches = connection.execute(
                f"SELECT {_FRIEND_COLUMNS} FROM friends "
                "WHERE name = ? COLLATE NOCASE ORDER BY id",
                (key,),
            ).fetchall()
            if len(matches) == 1:
                return dict(matches[0])
            if len(matches) > 1:
                raise AmbiguousFriendError(
                    key,
                    [int(row["id"]) for row in matches],
                )
            return None
        finally:
            connection.close()

    def list_friends(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return canonical contacts in deterministic name/id order."""
        normalized_limit = self._normalize_limit(limit)
        connection = connect(self.db_path, readonly=True)
        try:
            rows = connection.execute(
                f"SELECT {_FRIEND_COLUMNS} FROM friends "
                "ORDER BY name COLLATE NOCASE, id LIMIT ?",
                (normalized_limit,),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            connection.close()

    def search_friends(
        self,
        query: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search local canonical names without treating input as a LIKE pattern."""
        normalized_query = str(query).strip()
        if not normalized_query:
            return []

        escaped_query = (
            normalized_query.replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        normalized_limit = self._normalize_limit(limit)
        connection = connect(self.db_path, readonly=True)
        try:
            rows = connection.execute(
                f"SELECT {_FRIEND_COLUMNS} FROM friends "
                "WHERE name LIKE ? ESCAPE '\\' COLLATE NOCASE "
                "ORDER BY name COLLATE NOCASE, id LIMIT ?",
                (f"%{escaped_query}%", normalized_limit),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            connection.close()

    @staticmethod
    def _normalize_limit(limit: int) -> int:
        normalized = int(limit)
        if normalized < 1:
            raise ValueError("limit must be positive")
        return min(normalized, 500)
