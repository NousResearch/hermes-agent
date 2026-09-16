"""Tests for the Nomad service overlay (cron interflow graph).

A Nomad job self-declares its dataflow via ``hermes_*`` meta keys; the overlay
reads them with read-only ``nomad job`` CLI calls and emits graph nodes, gated
on a RUNNING allocation. The command runner is injected so the meta→declaration
+ liveness logic is exercised without a live Nomad agent.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _make_runner(jobs, meta_by_id=None, allocs_by_id=None):
    """Fake nomad CLI: `job status` lists jobs, `inspect` returns the spec's
    Meta, `allocs` returns allocations. All default to empty."""
    meta_by_id = meta_by_id or {}
    allocs_by_id = allocs_by_id or {}

    def runner(args):
        if args[:3] == ["nomad", "job", "status"]:
            return json.dumps(jobs)
        if args[:3] == ["nomad", "job", "inspect"]:
            job_id = args[-1]
            return json.dumps({"Job": {"ID": job_id}, "Meta": meta_by_id.get(job_id)})
        if args[:3] == ["nomad", "job", "allocs"]:
            job_id = args[-1]
            return json.dumps(allocs_by_id.get(job_id, []))
        raise AssertionError(f"unexpected nomad call: {args}")

    return runner


class TestSplitRefs:
    def test_splits_on_comma_and_whitespace_and_drops_blanks(self):
        from tools.nomad_services import _split_refs

        assert _split_refs("postgres:a, postgres:b   wiki:c") == [
            "postgres:a", "postgres:b", "wiki:c",
        ]
        assert _split_refs("") == []
        assert _split_refs(None) == []


class TestParseMeta:
    def test_valid_meta_becomes_declaration(self):
        from tools.nomad_services import _parse_meta_to_declaration

        decl = _parse_meta_to_declaration("honcho", {
            "hermes_service": "Honcho Memory API",
            "hermes_description": "# Honcho\nDialectic memory server.",
            "hermes_inputs": "postgres:honcho.sessions",
            "hermes_relationships": json.dumps([
                {"predicate": "supervised_by", "object": "scheduler:nomad"},
            ]),
        })
        assert decl == {
            "id": "nomad:honcho",
            "label": "Honcho Memory API",
            "description": "# Honcho\nDialectic memory server.",
            "inputs": ["postgres:honcho.sessions"],
            "outputs": [],
            "side_effects": [],
            "source_files": [],
            "relationships": [
                {"predicate": "supervised_by", "object": "scheduler:nomad"},
            ],
        }

    def test_source_files_meta_becomes_browsable_paths(self):
        # Nomad meta rejects dots, so `hermes_source_files` carries the flat
        # ref list; it rides onto the node so the alloc's code is browsable.
        from tools.nomad_services import _parse_meta_to_declaration

        decl = _parse_meta_to_declaration("honcho", {
            "hermes_service": "Honcho Memory API",
            "hermes_description": "serves memory context",
            "hermes_source_files": "app/server.py app/routes/",
        })
        # Normalization dedupes and sorts, exactly like a cron's source_files.
        assert decl["source_files"] == ["app/routes/", "app/server.py"]

    def test_no_service_meta_is_not_a_hermes_service(self):
        from tools.nomad_services import _parse_meta_to_declaration

        assert _parse_meta_to_declaration("j", {"owner": "x"}) is None
        assert _parse_meta_to_declaration("j", None) is None

    def test_missing_description_is_dropped(self):
        from tools.nomad_services import _parse_meta_to_declaration

        assert _parse_meta_to_declaration("j", {"hermes_service": "X"}) is None

    def test_bad_scheme_is_dropped(self):
        from tools.nomad_services import _parse_meta_to_declaration

        assert _parse_meta_to_declaration("j", {
            "hermes_service": "X",
            "hermes_description": "d",
            "hermes_outputs": "telegram:me",  # not an output scheme
        }) is None


class TestCollectNomadServices:
    def _job(self, jid, type_="service", status="running"):
        return {"ID": jid, "Type": type_, "Status": status}

    def test_collects_running_jobs_with_running_alloc(self):
        from tools.nomad_services import collect_nomad_services

        runner = _make_runner(
            jobs=[self._job("honcho"), self._job("other")],
            meta_by_id={
                "honcho": {
                    "hermes_service": "Honcho",
                    "hermes_description": "memory API",
                    "hermes_inputs": "postgres:honcho.sessions",
                },
                # declares but never inspected-failing: valid, listed
                "other": {},
            },
            allocs_by_id={"honcho": [{"ClientStatus": "running"}]},
        )
        services = collect_nomad_services(runner=runner)
        assert [s["label"] for s in services] == ["Honcho"]
        assert services[0]["health"]["status"] == "unknown"
        assert services[0]["health"]["probe"] == "nomad-allocation"
        assert services[0]["inputs"] == ["postgres:honcho.sessions"]

    def test_dead_allocation_is_not_live(self):
        from tools.nomad_services import collect_nomad_services

        runner = _make_runner(
            jobs=[self._job("honcho")],
            meta_by_id={"honcho": {
                "hermes_service": "Honcho",
                "hermes_description": "memory API",
            }},
            # desired running, but the allocation failed — not live
            allocs_by_id={"honcho": [{"ClientStatus": "failed"}]},
        )
        assert collect_nomad_services(runner=runner) == []

    def test_no_allocations_is_not_live(self):
        from tools.nomad_services import collect_nomad_services

        runner = _make_runner(
            jobs=[self._job("honcho")],
            meta_by_id={"honcho": {
                "hermes_service": "Honcho",
                "hermes_description": "memory API",
            }},
            allocs_by_id={"honcho": []},
        )
        assert collect_nomad_services(runner=runner) == []

    def test_batch_and_non_running_jobs_skipped(self):
        from tools.nomad_services import collect_nomad_services

        runner = _make_runner(
            jobs=[
                self._job("batchjob", type_="batch"),
                self._job("dead", status="dead"),
            ],
            meta_by_id={
                "batchjob": {
                    "hermes_service": "B",
                    "hermes_description": "d",
                },
                "dead": {
                    "hermes_service": "D",
                    "hermes_description": "d",
                },
            },
            allocs_by_id={
                "batchjob": [{"ClientStatus": "running"}],
                "dead": [{"ClientStatus": "running"}],
            },
        )
        assert collect_nomad_services(runner=runner) == []

    def test_nomad_unavailable_returns_empty(self):
        from tools.nomad_services import collect_nomad_services

        def boom(args):
            raise FileNotFoundError("nomad not installed")

        assert collect_nomad_services(runner=boom) == []

    def test_non_list_status_returns_empty(self):
        from tools.nomad_services import collect_nomad_services

        def runner(args):
            if args[:3] == ["nomad", "job", "status"]:
                return json.dumps({"error": "500 no servers"})
            raise AssertionError(f"unexpected call: {args}")

        assert collect_nomad_services(runner=runner) == []

    def test_overlay_links_nomad_service_to_cron_via_shared_store(self):
        # End-to-end through the graph builder: a Nomad-hosted API reading a
        # table a cron writes converges on the shared postgres node.
        from cron.jobs import build_cron_graph
        from tools.nomad_services import collect_nomad_services

        runner = _make_runner(
            jobs=[self._job("honcho")],
            meta_by_id={"honcho": {
                "hermes_service": "Honcho Memory API",
                "hermes_description": "reads sessions",
                "hermes_inputs": "postgres:honcho.sessions",
            }},
            allocs_by_id={"honcho": [{"ClientStatus": "running"}]},
        )
        jobs = [{
            "id": "job-index",
            "name": "index",
            "outputs": ["postgres:honcho.sessions"],
        }]
        graph = build_cron_graph(
            jobs=jobs, services=collect_nomad_services(runner=runner)
        )
        stores = [n for n in graph["nodes"] if n["id"] == "postgres:honcho.sessions"]
        assert len(stores) == 1 and stores[0]["kind"] == "artifact"
        assert {
            "source": "postgres:honcho.sessions",
            "target": "nomad:honcho",
            "type": "reads",
        } in graph["edges"]
        assert {
            "source": "job-index",
            "target": "postgres:honcho.sessions",
            "type": "writes",
        } in graph["edges"]
