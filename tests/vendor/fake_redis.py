"""In-memory Redis stand-in for harness AI-Scientist E2E tests."""

from __future__ import annotations


class FakeRedis:
    """Minimal LIST operations used by ai_scientist_runner / redis_loop."""

    def __init__(self) -> None:
        self.lists: dict[str, list[str]] = {}

    def ping(self) -> bool:
        return True

    def rpush(self, key: str, value: str) -> int:
        bucket = self.lists.setdefault(key, [])
        bucket.append(value)
        return len(bucket)

    def lpush(self, key: str, value: str) -> int:
        bucket = self.lists.setdefault(key, [])
        bucket.insert(0, value)
        return len(bucket)

    def ltrim(self, key: str, start: int, end: int) -> bool:
        bucket = self.lists.get(key, [])
        if not bucket:
            return True
        length = len(bucket)
        if start < 0:
            start = max(length + start, 0)
        if end < 0:
            end = length + end
        self.lists[key] = bucket[start : end + 1]
        return True

    def llen(self, key: str) -> int:
        return len(self.lists.get(key, []))

    def lrange(self, key: str, start: int, end: int) -> list[str]:
        bucket = self.lists.get(key, [])
        if not bucket:
            return []
        length = len(bucket)
        if start < 0:
            start = max(length + start, 0)
        if end < 0:
            end = length + end
        return bucket[start : end + 1]

    def lpop(self, key: str) -> str | None:
        bucket = self.lists.get(key, [])
        if not bucket:
            return None
        return bucket.pop(0)

    def exists(self, key: str) -> bool:
        return key in self.lists and bool(self.lists[key])

    def get(self, key: str) -> str | None:
        bucket = self.lists.get(key, [])
        return bucket[-1] if bucket else None
