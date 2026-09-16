"""Tests for the Docker service overlay (cron interflow graph).

A Docker container self-declares its dataflow via ``hermes.*`` labels; the
overlay reads them with a read-only ``docker ps``/``docker inspect`` and emits
graph nodes. The command runner is injected so the label→declaration logic is
exercised without a live Docker daemon.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _make_runner(ids, labels_by_id, health_by_id=None):
    """Fake docker CLI: `ps` lists ids, `inspect` returns a container's labels."""
    def runner(args):
        if args[:2] == ["docker", "ps"]:
            return "\n".join(ids) + "\n"
        if args[:2] == ["docker", "inspect"]:
            if ".State.Health.Status" in args[3]:
                return (health_by_id or {}).get(args[-1], "")
            return json.dumps(labels_by_id.get(args[-1]))
        raise AssertionError(f"unexpected docker call: {args}")
    return runner


class TestSplitRefs:
    def test_splits_on_comma_and_whitespace_and_drops_blanks(self):
        from tools.docker_services import _split_refs

        assert _split_refs("postgres:a, postgres:b   wiki:c") == [
            "postgres:a", "postgres:b", "wiki:c",
        ]
        assert _split_refs("") == []
        assert _split_refs(None) == []


class TestParseLabels:
    def test_valid_labels_become_declaration(self):
        from tools.docker_services import _parse_labels_to_declaration

        decl = _parse_labels_to_declaration("abcdef012345678", {
            "hermes.service": "Analytics Dashboard",
            "hermes.description": "# Dash\nRenders analytics.",
            "hermes.inputs": "postgres:analytics.events",
            "hermes.relationships": json.dumps([
                {"predicate": "runs_in", "object": "runtime:docker"},
            ]),
        })
        assert decl == {
            "id": "docker:abcdef012345",  # truncated to 12
            "label": "Analytics Dashboard",
            "description": "# Dash\nRenders analytics.",
            "inputs": ["postgres:analytics.events"],
            "outputs": [],
            "side_effects": [],
            "source_files": [],
            "relationships": [
                {"predicate": "runs_in", "object": "runtime:docker"},
            ],
        }

    def test_source_files_label_becomes_browsable_paths(self):
        # `hermes.source_files` is a flat comma/space list like the other refs;
        # it rides onto the node so the container's code is browsable in the graph.
        from tools.docker_services import _parse_labels_to_declaration

        decl = _parse_labels_to_declaration("abcdef012345678", {
            "hermes.service": "Dashboard",
            "hermes.description": "serves the app",
            "hermes.source_files": "app/server.py app/routes/",
        })
        # Normalization dedupes and sorts, exactly like a cron's source_files.
        assert decl["source_files"] == ["app/routes/", "app/server.py"]

    def test_no_service_label_is_not_a_hermes_service(self):
        from tools.docker_services import _parse_labels_to_declaration

        assert _parse_labels_to_declaration("abc", {"maintainer": "x"}) is None
        assert _parse_labels_to_declaration("abc", None) is None

    def test_missing_description_is_dropped(self):
        from tools.docker_services import _parse_labels_to_declaration

        # A tracked service with no description is invalid — dropped, not raised.
        assert _parse_labels_to_declaration("abc", {"hermes.service": "X"}) is None

    def test_bad_scheme_is_dropped(self):
        from tools.docker_services import _parse_labels_to_declaration

        assert _parse_labels_to_declaration("abc", {
            "hermes.service": "X",
            "hermes.description": "d",
            "hermes.inputs": "telegram:me",  # not an input scheme
        }) is None


class TestCollectDockerServices:
    def test_exposes_native_container_health(self):
        from tools.docker_services import collect_docker_services

        services = collect_docker_services(runner=_make_runner(
            ids=["c1"],
            labels_by_id={"c1": {
                "hermes.service": "API",
                "hermes.description": "serves requests",
            }},
            health_by_id={"c1": "unhealthy"},
        ))

        assert services[0]["health"] == {
            "status": "unhealthy",
            "probe": "docker-healthcheck",
            "target": "docker:c1",
            "checked_at": "",
            "latency_ms": 0,
            "message": "Docker healthcheck: unhealthy",
        }

    def test_collects_running_labeled_containers(self):
        from tools.docker_services import collect_docker_services

        runner = _make_runner(
            ids=["c1", "c2"],
            labels_by_id={
                "c1": {
                    "hermes.service": "Dashboard",
                    "hermes.description": "reads events",
                    "hermes.inputs": "postgres:analytics.events",
                },
                "c2": {  # invalid: no description → dropped
                    "hermes.service": "Broken",
                },
            },
        )
        services = collect_docker_services(runner=runner)
        assert [s["label"] for s in services] == ["Dashboard"]
        assert services[0]["id"] == "docker:c1"
        assert services[0]["inputs"] == ["postgres:analytics.events"]

    def test_docker_unavailable_returns_empty(self):
        from tools.docker_services import collect_docker_services

        def boom(args):
            raise FileNotFoundError("docker not installed")

        assert collect_docker_services(runner=boom) == []

    def test_overlay_links_docker_service_to_cron_via_shared_store(self):
        # End-to-end through the graph builder: a Docker dashboard reading a
        # table a cron writes converges on the shared postgres node.
        from cron.jobs import build_cron_graph
        from tools.docker_services import collect_docker_services

        runner = _make_runner(
            ids=["c1"],
            labels_by_id={"c1": {
                "hermes.service": "Dashboard",
                "hermes.description": "reads events",
                "hermes.inputs": "postgres:analytics.events",
            }},
        )
        jobs = [{
            "id": "job-ingest",
            "name": "ingest",
            "outputs": ["postgres:analytics.events"],
        }]
        graph = build_cron_graph(
            jobs=jobs, services=collect_docker_services(runner=runner)
        )
        stores = [n for n in graph["nodes"] if n["id"] == "postgres:analytics.events"]
        assert len(stores) == 1 and stores[0]["kind"] == "artifact"
        assert {
            "source": "postgres:analytics.events",
            "target": "docker:c1",
            "type": "reads",
        } in graph["edges"]
        assert {
            "source": "job-ingest",
            "target": "postgres:analytics.events",
            "type": "writes",
        } in graph["edges"]
