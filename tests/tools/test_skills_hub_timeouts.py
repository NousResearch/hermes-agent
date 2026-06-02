import time

from tools.skills_hub import SkillMeta, SkillSource, parallel_search_sources


class _FastSource(SkillSource):
    def source_id(self):
        return "fast"

    def search(self, query, limit=10):
        return [
            SkillMeta(
                name="fast-skill",
                description="fast",
                source="fast",
                identifier="fast/skill",
                trust_level="community",
            )
        ]

    def fetch(self, identifier):
        return None

    def inspect(self, identifier):
        return None


class _StuckSource(SkillSource):
    def source_id(self):
        return "stuck"

    def search(self, query, limit=10):
        time.sleep(1.5)
        return []

    def fetch(self, identifier):
        return None

    def inspect(self, identifier):
        return None


def test_parallel_search_returns_promptly_after_timeout_with_partial_results():
    start = time.monotonic()

    results, counts, timed_out = parallel_search_sources(
        [_FastSource(), _StuckSource()],
        query="demo",
        overall_timeout=0.05,
    )

    elapsed = time.monotonic() - start
    assert elapsed < 0.5
    assert [result.name for result in results] == ["fast-skill"]
    assert counts == {"fast": 1}
    assert timed_out == ["stuck"]
