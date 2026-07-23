"""Application-facing read operations for Facebook automation."""

from __future__ import annotations

from typing import Any

from .repository import AmbiguousFriendError, FacebookRepository


class FacebookApplicationService:
    """Coordinate stable read-only use cases over the canonical repository."""

    def __init__(self, repository: FacebookRepository | None = None) -> None:
        self.repository = repository or FacebookRepository()

    def show_friend(self, friend_key: str | int) -> dict[str, Any]:
        try:
            friend = self.repository.find_friend(friend_key)
        except AmbiguousFriendError as exc:
            return {
                "success": False,
                "error": "Ambiguous friend name; use CRM id",
                "matching_ids": list(exc.matching_ids),
            }
        if friend is None:
            return {
                "success": False,
                "error": f"Friend '{friend_key}' not found in database",
            }
        return {"success": True, "friend": friend}

    def list_friends(self, limit: int = 50) -> dict[str, Any]:
        friends = self.repository.list_friends(limit)
        return {
            "success": True,
            "count": len(friends),
            "friends": friends,
        }

    def search_friends(self, query: str, limit: int = 20) -> dict[str, Any]:
        normalized_query = str(query).strip()
        if not normalized_query:
            return {"success": False, "error": "Search query must not be empty"}

        friends = self.repository.search_friends(normalized_query, limit)
        return {
            "success": True,
            "query": normalized_query,
            "results_count": len(friends),
            "friends": friends,
        }
