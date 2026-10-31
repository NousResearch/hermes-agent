"""Tests for GitHub tap listing of author-published multi-skill repos.

Covers the repo-root tap shape (``path: ""``, the garrytan/gstack pattern):
`_list_skills_in_repo` lists every root-level skill directory and stamps no
category when the repo ships no skills.sh.json grouping. Companion to the
`justinjohnson25600/hermes_skills` DEFAULT_TAPS entry.
"""

from unittest.mock import patch, MagicMock

from tools.skills_hub_github import GitHubSource
from tools.skills_hub_models import SkillMeta


def test_list_skills_root_path_tap_without_bucket():
    # A tap with path="" (repo root as the skill directory) lists every root
    # level skill dir and stamps no category — the garrytan/gstack shape.
    src = GitHubSource(auth=MagicMock())
    meta = SkillMeta(
        name="dev-pair", description="d", source="github",
        identifier="justinjohnson25600/hermes_skills/dev-pair", trust_level="community",
    )
    resp = MagicMock(status_code=200)
    resp.json.return_value = [{"type": "dir", "name": "dev-pair"}]
    with patch("tools.skills_hub_github._cached_metas", return_value=None), \
         patch("tools.skills_hub_github._cache_metas"), \
         patch.object(src, "_get_skillsh_groupings", return_value=None), \
         patch.object(src, "inspect", return_value=meta), \
         patch.object(src, "_github_get", return_value=resp):
        skills = src._list_skills_in_repo("justinjohnson25600/hermes_skills", "")

    assert len(skills) == 1
    assert skills[0].name == "dev-pair"
    assert skills[0].identifier == "justinjohnson25600/hermes_skills/dev-pair"
    assert "category" not in skills[0].extra


def test_tapped_repo_present_in_default_taps():
    # The tap entry must exist so the repo's skills are browsable without any
    # user-side `hermes skills tap add` (mirrors the TRUSTED_REPOS invariant
    # test's approach of asserting membership, not path equality).
    tap_repos = {tap["repo"] for tap in GitHubSource.DEFAULT_TAPS}
    assert "justinjohnson25600/hermes_skills" in tap_repos
