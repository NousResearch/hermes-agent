"""Tests for the launchd service overlay (cron interflow graph).

A launchd service's dataflow declaration lives in a sidecar JSON under
``~/.hermes/services/launchd/`` (a plist has no label store); liveness is
probed via ``launchctl print`` state. Runner and registry dir are injected so
the sidecar→declaration + probe logic is exercised without a live launchd.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _sidecar(label, name=None, description="serves memory context", **extra):
    doc = {
        "label": label,
        "name": name or label,
        "description": description,
    }
    doc.update(extra)
    return doc


def _make_registry(tmp_path, docs_by_filename):
    for filename, doc in docs_by_filename.items():
        p = tmp_path / filename
        p.write_text(json.dumps(doc), encoding="utf-8")
    return tmp_path


def _running_probe(labels_running):
    """Fake launchctl: `print gui/<uid>/<label>` returns a state key-dump."""

    def runner(args):
        if args[:2] == ["launchctl", "print"]:
            label = args[-1].rsplit("/", 1)[-1]
            if label in labels_running:
                return (
                    "services:\n"
                    "\tstate = running\n"
                    "\tprogram = /bin/foo\n"
                )
            raise subprocess_error(f"not loaded: {label}")
        raise AssertionError(f"unexpected launchctl call: {args}")

    return runner


def subprocess_error(msg):
    import subprocess

    return subprocess.CalledProcessError(1, ["launchctl"], msg)


class TestParseSidecar:
    def test_valid_sidecar_becomes_declaration(self):
        from tools.launchd_services import _parse_sidecar_to_declaration

        path = Path("/fake/dev.redis.plist-label.json")  # stem == label below
        path = Path(f"/fake/{_safe_stem('dev.redis')}.json")
        decl = _parse_sidecar_to_declaration(path, _sidecar(
            "dev.redis", name="Redis (brew)",
            inputs=["file:/opt/homebrew/etc/redis.conf"],
            relationships=[
                {"predicate": "supervised_by", "object": "scheduler:launchd"},
            ],
        ))
        assert decl == {
            "id": "launchd:dev.redis",
            "label": "Redis (brew)",
            "description": "serves memory context",
            "inputs": ["file:/opt/homebrew/etc/redis.conf"],
            "outputs": [],
            "side_effects": [],
            "source_files": [],
            "relationships": [
                {"predicate": "supervised_by", "object": "scheduler:launchd"},
            ],
        }

    def test_source_files_sidecar_becomes_browsable_paths(self):
        # The launchd sidecar is structured JSON, so `source_files` is a real
        # list; it rides onto the node so the service's code is browsable.
        from tools.launchd_services import _parse_sidecar_to_declaration

        path = Path(f"/fake/{_safe_stem('dev.redis')}.json")
        decl = _parse_sidecar_to_declaration(path, _sidecar(
            "dev.redis", name="Redis (brew)",
            source_files=["scripts/redis-wrap.sh", "app/routes/"],
        ))
        assert decl["source_files"] == ["app/routes/", "scripts/redis-wrap.sh"]

    def test_label_mismatch_is_dropped(self):
        from tools.launchd_services import _parse_sidecar_to_declaration

        path = Path("/fake/dev.redis.json")
        # file stem is dev.redis but the doc declares another label
        assert _parse_sidecar_to_declaration(
            path, _sidecar("com.other.service")
        ) is None

    def test_missing_label_is_dropped(self):
        from tools.launchd_services import _parse_sidecar_to_declaration

        assert _parse_sidecar_to_declaration(
            Path("/fake/x.json"), {"name": "X", "description": "d"}
        ) is None

    def test_missing_description_is_dropped(self):
        from tools.launchd_services import _parse_sidecar_to_declaration

        assert _parse_sidecar_to_declaration(
            Path("/fake/x.json"), {"label": "x", "name": "X"}
        ) is None

    def test_non_dict_is_dropped(self):
        from tools.launchd_services import _parse_sidecar_to_declaration

        doc = ["nope"]  # type: ignore[arg-type]  # deliberately wrong shape
        assert _parse_sidecar_to_declaration(Path("/fake/x.json"), doc) is None

    def test_bad_scheme_is_dropped(self):
        from tools.launchd_services import _parse_sidecar_to_declaration

        path = Path("/fake/x.json")
        assert _parse_sidecar_to_declaration(path, _sidecar(
            "x", outputs=["telegram:me"]  # not an output scheme
        )) is None


def _safe_stem(label):
    return label  # filename stem equals the launchd label


class TestCollectLaunchdServices:
    def test_running_service_uses_declared_application_health_probe(self, tmp_path):
        from tools.launchd_services import collect_launchd_services

        registry = _make_registry(tmp_path, {
            "dev.api.json": _sidecar(
                "dev.api",
                service_health={
                    "type": "http",
                    "url": "http://127.0.0.1:8080/health",
                    "expected_status": 204,
                    "timeout_seconds": 1,
                },
            ),
        })
        seen = []

        def health_prober(spec):
            seen.append(spec)
            return {
                "status": "healthy",
                "probe": "http",
                "target": spec["url"],
                "checked_at": "2026-09-04T00:00:00Z",
                "latency_ms": 2.5,
                "message": "HTTP 204",
            }

        services = collect_launchd_services(
            runner=_running_probe({"dev.api"}),
            registry_dir=registry,
            health_prober=health_prober,
        )

        assert seen == [{
            "type": "http",
            "url": "http://127.0.0.1:8080/health",
            "expected_status": 204,
            "timeout_seconds": 1.0,
            "startup_timeout_seconds": 30.0,
        }]
        assert services[0]["health"] == {
            "status": "healthy",
            "probe": "http",
            "target": "http://127.0.0.1:8080/health",
            "checked_at": "2026-09-04T00:00:00Z",
            "latency_ms": 2.5,
            "message": "HTTP 204",
        }
        assert "_health_spec" not in services[0]
        from cron.jobs import build_cron_graph

        graph = build_cron_graph(jobs=[], services=services)
        node = next(n for n in graph["nodes"] if n["id"] == "launchd:dev.api")
        assert node["health"] == services[0]["health"]
        assert "_health_spec" not in node

    def test_application_probe_failure_is_health_evidence_not_graph_failure(self, tmp_path):
        from tools.launchd_services import collect_launchd_services

        registry = _make_registry(tmp_path, {
            "dev.api.json": _sidecar(
                "dev.api",
                service_health={
                    "type": "http",
                    "url": "http://127.0.0.1:8080/health",
                },
            ),
        })

        def health_prober(_spec):
            raise RuntimeError("probe implementation failed")

        services = collect_launchd_services(
            runner=_running_probe({"dev.api"}),
            registry_dir=registry,
            health_prober=health_prober,
        )

        assert len(services) == 1
        assert services[0]["health"]["status"] == "unhealthy"
        assert services[0]["health"]["probe"] == "http"
        assert services[0]["health"]["target"] == "http://127.0.0.1:8080/health"
        assert services[0]["health"]["message"] == (
            "RuntimeError: probe implementation failed"
        )

    def test_collects_registered_and_running(self, tmp_path):
        from tools.launchd_services import collect_launchd_services

        registry = _make_registry(tmp_path, {
            "dev.redis.json": _sidecar(
                "dev.redis", name="Redis (brew)",
                inputs=["file:/opt/homebrew/etc/redis.conf"],
            ),
        })
        services = collect_launchd_services(
            runner=_running_probe({"dev.redis"}),
            registry_dir=registry,
        )
        assert [s["id"] for s in services] == ["launchd:dev.redis"]
        assert services[0]["label"] == "Redis (brew)"
        assert services[0]["health"]["status"] == "unknown"
        assert services[0]["health"]["probe"] == "launchctl"

    def test_invalid_application_health_contract_drops_sidecar(self, tmp_path):
        from tools.launchd_services import collect_launchd_services

        registry = _make_registry(tmp_path, {
            "dev.api.json": _sidecar(
                "dev.api",
                service_health={"type": "http", "url": "file:///tmp/healthy"},
            ),
        })

        assert collect_launchd_services(
            runner=_running_probe({"dev.api"}),
            registry_dir=registry,
        ) == []

    def test_registered_but_not_running_is_skipped(self, tmp_path):
        from tools.launchd_services import collect_launchd_services

        registry = _make_registry(tmp_path, {
            "dev.redis.json": _sidecar(
                "dev.redis",
                service_health={
                    "type": "http",
                    "url": "http://127.0.0.1:8080/health",
                },
            ),
        })

        def forbidden_health_probe(_spec):
            raise AssertionError("inactive launchd service must not be probed")

        # probe raises (not loaded) → not running
        services = collect_launchd_services(
            runner=_running_probe(set()),
            registry_dir=registry,
            health_prober=forbidden_health_probe,
        )
        assert services == []

    def test_state_not_running_is_skipped(self, tmp_path):
        from tools.launchd_services import collect_launchd_services

        registry = _make_registry(tmp_path, {
            "dev.redis.json": _sidecar("dev.redis"),
        })

        def runner(args):
            if args[:2] == ["launchctl", "print"]:
                return "state = waiting\n"  # loaded but not in run loop
            raise AssertionError(f"unexpected call: {args}")

        assert collect_launchd_services(
            runner=runner, registry_dir=registry
        ) == []

    def test_malformed_json_dropped_not_raised(self, tmp_path):
        from tools.launchd_services import collect_launchd_services

        (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
        (tmp_path / "good.json").write_text(
            json.dumps(_sidecar("good")), encoding="utf-8"
        )
        services = collect_launchd_services(
            runner=_running_probe({"good"}), registry_dir=tmp_path
        )
        assert [s["id"] for s in services] == ["launchd:good"]

    def test_missing_registry_returns_empty(self, tmp_path):
        from tools.launchd_services import collect_launchd_services

        services = collect_launchd_services(
            runner=_running_probe(set()),
            registry_dir=tmp_path / "does-not-exist",
        )
        assert services == []

    def test_launchctl_unavailable_drops_all(self, tmp_path):
        from tools.launchd_services import collect_launchd_services

        registry = _make_registry(tmp_path, {
            "dev.redis.json": _sidecar("dev.redis"),
        })

        def boom(args):
            raise FileNotFoundError("launchctl missing")

        assert collect_launchd_services(
            runner=boom, registry_dir=registry
        ) == []

    def test_overlay_links_launchd_service_to_cron_via_shared_store(self, tmp_path):
        # End-to-end through the graph builder: a brew Postgres hosting a table
        # a cron writes converges on the shared postgres node.
        from cron.jobs import build_cron_graph
        from tools.launchd_services import collect_launchd_services

        registry = _make_registry(tmp_path, {
            "dev.postgresql@17.json": _sidecar(
                "dev.postgresql@17", name="PostgreSQL 17 (brew)",
                inputs=["postgres:analytics.events"],
            ),
        })
        jobs = [{
            "id": "job-ingest",
            "name": "ingest",
            "outputs": ["postgres:analytics.events"],
        }]
        graph = build_cron_graph(
            jobs=jobs,
            services=collect_launchd_services(
                runner=_running_probe({"dev.postgresql@17"}),
                registry_dir=registry,
            ),
        )
        stores = [n for n in graph["nodes"] if n["id"] == "postgres:analytics.events"]
        assert len(stores) == 1 and stores[0]["kind"] == "artifact"
        assert {
            "source": "postgres:analytics.events",
            "target": "launchd:dev.postgresql@17",
            "type": "reads",
        } in graph["edges"]
        assert {
            "source": "job-ingest",
            "target": "postgres:analytics.events",
            "type": "writes",
        } in graph["edges"]
