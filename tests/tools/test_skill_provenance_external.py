"""#108032: external-dir skills get an 'external' provenance tier, not 'agent'.

Externally mounted skills (skills.external_dirs) are externally authored: the label must say
where they came from without changing their foreground mutability (in-place edits are the
standing design, commit 8c8fc6c1ec / PR #17512) and without counting as learning milestones
in the journey graph.
"""

import pytest


@pytest.fixture
def external_home(tmp_path, monkeypatch):
    """HERMES_HOME with one external dir configured and one external skill in it.

    The external skill is ALSO reachable from the profile skills tree via a symlinked
    SKILL.md — the common mounting style — because that is the shape the journey graph
    actually sees (its roots are the bundled and profile skills dirs; rglob there does not
    traverse dir symlinks but does yield symlinked files, and is_external_skill_path
    resolves the link to tag it 'external').
    """
    home = tmp_path / ".hermes"
    (home / "skills").mkdir(parents=True)
    ext_dir = tmp_path / "vault"
    (ext_dir / "ext-skill").mkdir(parents=True)
    (ext_dir / "ext-skill" / "SKILL.md").write_text(
        "---\nname: ext-skill\ndescription: An external skill.\n---\n\n# External\n", encoding="utf-8")
    (home / "skills" / "ext-linked").mkdir()
    (home / "skills" / "ext-linked" / "SKILL.md").symlink_to(ext_dir / "ext-skill" / "SKILL.md")
    (home / "config.yaml").write_text(f"skills:\n  external_dirs:\n    - {ext_dir}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    from agent import skill_utils
    skill_utils._external_dirs_cache_clear()
    yield home
    skill_utils._external_dirs_cache_clear()


def _write_local_skill(home, name: str):
    d = home / "skills" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Local skill.\n---\n\n# Local\n", encoding="utf-8")


class TestProvenanceTier:
    def test_external_only_skill_classifies_as_external(self, external_home):
        from tools.skill_usage import provenance
        assert provenance("ext-skill") == "external"

    def test_local_skill_stays_agent(self, external_home):
        _write_local_skill(external_home, "my-local")
        from tools.skill_usage import provenance
        assert provenance("my-local") == "agent"

    def test_local_copy_wins_over_external_mount(self, external_home):
        """A name present both locally and externally is NOT external (local store wins)."""
        _write_local_skill(external_home, "ext-skill")
        from tools.skill_usage import provenance
        assert provenance("ext-skill") == "agent"

    def test_agent_created_still_excludes_external_only(self, external_home):
        from tools.skill_usage import is_agent_created
        assert is_agent_created("ext-skill") is False

    def test_external_names_helper_scans_external_dirs(self, external_home):
        from tools.skill_usage import _external_skill_names
        assert _external_skill_names() == {"ext-skill"}


class TestLearningGraphExcludesExternal:
    def test_used_external_skill_produces_no_learning_node(self, external_home):
        """Even a USED external skill is not a learning milestone (#108032)."""
        from agent import learning_graph
        with _patched_usage({"ext-skill": {"use_count": 5, "created_by": None}}):
            skill_ids = {n["id"] for n in learning_graph.build_learning_graph()["nodes"]
                         if n["kind"] == "skill"}
        assert "ext-skill" not in skill_ids

    def test_used_local_skill_still_produces_learning_node(self, external_home):
        _write_local_skill(external_home, "my-local")
        from agent import learning_graph
        with _patched_usage({"my-local": {"use_count": 3, "created_by": None}}):
            skill_ids = {n["id"] for n in learning_graph.build_learning_graph()["nodes"]
                         if n["kind"] == "skill"}
        assert "my-local" in skill_ids

    def test_external_source_does_not_leak_to_later_skills_in_a_root(self, external_home, monkeypatch):
        """Order-forced twin of the above: when the external mount scans BEFORE the local
        skill in one rglob root, the local skill must still classify 'agent' — the loop
        must not carry 'external' forward on the ``source`` variable. The original leak only
        surfaced under filesystem layouts that happened to interleave the mount first, which
        is why it masqueraded as a runner/env-dependence (run_tests.sh vs bare pytest) rather
        than the scan-order bug it was."""
        from pathlib import Path
        from agent import learning_graph
        _write_local_skill(external_home, "my-local")
        real_rglob = Path.rglob

        def hostile_rglob(self, pattern):
            hits = sorted(real_rglob(self, pattern))
            # Every skill root: yield the external mount's SKILL.md BEFORE any local one.
            return sorted(hits, key=lambda p: 0 if "ext-" in p.parent.name else 1)

        monkeypatch.setattr(Path, "rglob", hostile_rglob)
        with _patched_usage({"my-local": {"use_count": 3, "created_by": None}}):
            graph = learning_graph.build_learning_graph()
        sources = {n["id"]: n.get("source") for n in graph["nodes"] if n["kind"] == "skill"}
        assert "my-local" in sources and sources["my-local"] != "external"


def _patched_usage(data):
    from unittest.mock import patch
    from agent import learning_graph
    return patch.object(learning_graph, "_load_usage", lambda: data)


class TestWebRouterProvenanceTwin:
    """The /api/skills set-based twin must classify external mounts as 'external' while a
    local hand-made skill stays 'agent' (the regression this guards: hub > bundled > agent
    with no external tier flattened mounts into 'agent').

    ``get_skills`` resolves its imports inside the function body, so monkeypatching router
    module attributes cannot intercept them; drive the real app through TestClient instead
    (same harness as test_web_server_skills_profiles.py)."""

    @pytest.fixture
    def client(self, external_home, monkeypatch):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        import hermes_state
        from hermes_constants import get_hermes_home
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
        c = TestClient(app)
        c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
        return c

    def test_router_labels_external_vs_agent(self, client, external_home):
        _write_local_skill(external_home, "my-local")
        resp = client.get("/api/skills")
        assert resp.status_code == 200, resp.text
        by_name = {s["name"]: s["provenance"] for s in resp.json()}
        assert by_name.get("ext-skill") == "external"
        assert by_name.get("my-local") == "agent"


class TestRouterOriginTwin:
    """#70712: /api/skills must expose the explicit human-facing ``origin`` —
    a local unclassified skill is 'local' (never 'learned'), a background-review
    sediment is 'background_review' — while legacy ``provenance`` ownership stays
    'agent' for older clients. Same TestClient harness as the provenance twin."""

    def _client(self, external_home, monkeypatch):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")
        import hermes_state
        from hermes_constants import get_hermes_home
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
        c = TestClient(app)
        c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
        return c

    def test_router_labels_local_vs_background_review(self, external_home, monkeypatch):
        from tools.skill_usage import record_created

        _write_local_skill(external_home, "manual-skill")
        _write_local_skill(external_home, "review-sediment")
        record_created("review-sediment", agent_created=True)
        client = self._client(external_home, monkeypatch)
        resp = client.get("/api/skills")
        assert resp.status_code == 200, resp.text
        by_name = {s["name"]: s for s in resp.json()}
        # Legacy ownership: both stay editable 'agent' for older clients.
        assert by_name["manual-skill"]["provenance"] == "agent"
        assert by_name["review-sediment"]["provenance"] == "agent"
        # Explicit origin: only the background-review sediment is 'learned'.
        assert by_name["manual-skill"]["origin"] == "local"
        assert by_name["review-sediment"]["origin"] == "background_review"
        assert by_name["ext-skill"]["origin"] == "external"
