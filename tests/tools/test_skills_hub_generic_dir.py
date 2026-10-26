"""Skills hub: a skills.sh skill kept in a generic dir (``skills/SKILL.md``) resolves and installs under its slug."""

from unittest.mock import MagicMock, patch

from tools.skills_hub_github import GitHubAuth, GitHubSource
from tools.skills_hub_models import SkillBundle
from tools.skills_hub_official import HermesIndexSource
from tools.skills_hub_skillssh import SkillsShSource

ROW = {"name": "weread-skills", "description": "WeChat Reading skills", "source": "skills.sh",
       "identifier": "skills-sh/tencent/wechatreading/weread-skills",
       "repo": "tencent/wechatreading", "path": "weread-skills"}


def _generic_bundle(*_):
    return SkillBundle(name="skills", files={"SKILL.md": "---\nname: weread-skills\n---\n"},
                       source="github", identifier="tencent/wechatreading/skills", trust_level="community")


def _index_source(row, github, monkeypatch):
    src = HermesIndexSource(auth=GitHubAuth())
    src._index, src._loaded = {"skills": [row]}, True
    monkeypatch.setattr(src, "_get_github", lambda: github)
    return src


@patch("tools.skills_hub.httpx.get")
def test_finds_single_generic_skills_directory_when_slug_is_stale(mock_get):
    """A lone ``skills/SKILL.md`` answers for a stale slug; a lone NAMED skill (``skills/bar``) never does."""
    auth = MagicMock(spec=GitHubAuth)
    auth.get_headers.return_value = {"Accept": "application/vnd.github.v3+json"}
    for lone, expected in (("skills/SKILL.md", "tencent/wechatreading/skills"), ("skills/bar/SKILL.md", None)):
        tree_entries = [{"path": "README.md", "type": "blob"}, {"path": lone, "type": "blob"}]
        mock_get.side_effect = [
            MagicMock(status_code=200, json=lambda: {"default_branch": "main"}),
            MagicMock(status_code=200, json=lambda t=tree_entries: {"tree": t}),
        ]

        assert GitHubSource(auth=auth)._find_skill_in_repo_tree("tencent/wechatreading", "weread-skills") == expected


def test_fetch_resolves_generic_skill_directory_from_index_slug(monkeypatch):
    """A skills.sh skill under a generic ``skills/`` dir installs as its slug, whether the index fell
    back to the tree, resolved it, or skills.sh fetched it directly; a gone GitHub path stays gone."""
    github = MagicMock()
    github._find_skill_in_repo_tree.return_value = "tencent/wechatreading/skills"
    github._find_repo_root_skill.return_value = None
    for extra, fetched in (({}, [None, _generic_bundle()]),
                           ({"resolved_github_id": "tencent/wechatreading/skills"}, [_generic_bundle()])):
        github.fetch.side_effect = fetched

        bundle = _index_source({**ROW, **extra}, github, monkeypatch).fetch(ROW["identifier"])

        assert (bundle.name, bundle.identifier) == ("weread-skills", ROW["identifier"])  # one lock name
    github._find_skill_in_repo_tree.assert_called_once_with("tencent/wechatreading", "weread-skills")

    skills_sh = SkillsShSource(auth=MagicMock(spec=GitHubAuth))
    monkeypatch.setattr(skills_sh, "_fetch_detail_page", lambda _id: None)
    monkeypatch.setattr(skills_sh, "_candidate_identifiers", lambda _id: [])
    monkeypatch.setattr(skills_sh, "_discover_identifier", lambda *_a, **_k: "tencent/wechatreading/skills")
    monkeypatch.setattr(skills_sh.github, "fetch", _generic_bundle)
    assert skills_sh.fetch(ROW["identifier"]).name == "weread-skills"

    github.reset_mock()
    github.fetch.side_effect = [None]
    gone = {**ROW, "source": "github", "identifier": "tencent/wechatreading/gone", "path": "gone"}
    assert _index_source(gone, github, monkeypatch).fetch(gone["identifier"]) is None  # no other skill relabeled
    github._find_skill_in_repo_tree.assert_not_called()
