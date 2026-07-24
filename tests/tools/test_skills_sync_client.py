"""Tests for tools/skills_sync_client.py — the HSP/1 sync client.

Covers, against the frozen contract (~/src/specs/collective-wisdom/
hsp-1-contract.md):
  * content addressing (full 64-hex) + canonical JSON (§2.1, §2.5)
  * the DEV-PHASE gate (tool_gateway_admin) making sync inert
  * the M1-D opt-in default (nothing syncs without the sync flag)
  * object building (blob/tree/commit, exec mode, size limit)
  * push (upload + CAS), pull (materialize), and the three-way merge / 409
    conflict paths — all against an in-process mock HSP server.

The mock server implements the contract §3/§4 endpoint shapes with an
in-memory object store + ref table. No live server, no network.
"""

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

import tools.skills_sync_client as ssc


# ---------------------------------------------------------------------------
# In-process mock HSP/1 server (contract §3-§4)
# ---------------------------------------------------------------------------

class _MockState:
    def __init__(self):
        self.objects = {}   # hash -> (kind, bytes)
        self.refs = {}      # name -> commit hash
        self.hsp_version = "1"
        self.max_object_bytes = 26214400
        self.force_conflict_once = False  # inject a 409 on the next CAS
        # M2 org behavior (contract §11): advertise the "org" feature and,
        # when org_role_admin is False, convert org-HEAD CAS to 202 proposals.
        self.org_feature = True
        self.org_role_admin = True
        self.proposals = []  # [{n, to, base}]


def _make_handler(state: _MockState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # silence
            pass

        def _json(self, code, obj, extra_headers=None):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            for k, v in (extra_headers or {}).items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            query = ""
            if "?" in self.path:
                query = self.path.split("?", 1)[1]

            if path == "/v1/sync/capabilities":
                features = ["personal"] + (["org"] if state.org_feature else [])
                return self._json(200, {
                    "hsp_version": state.hsp_version,
                    "features": features,
                    "max_object_bytes": state.max_object_bytes,
                    "hash_alg": "sha256",
                    "auth": "bearer",
                })

            if path == "/v1/sync/refs":
                prefix = ""
                for part in query.split("&"):
                    if part.startswith("prefix="):
                        from urllib.parse import unquote
                        prefix = unquote(part[len("prefix="):])
                refs = [
                    {"name": n, "hash": h}
                    for n, h in state.refs.items()
                    if n.startswith(prefix)
                ]
                return self._json(200, {"refs": refs})

            if path.startswith("/v1/sync/objects/"):
                obj_hash = path[len("/v1/sync/objects/"):]
                if obj_hash not in state.objects:
                    return self._json(404, {"error": "not_found"})
                kind, data = state.objects[obj_hash]
                if kind == ssc.KIND_BLOB:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("X-HSP-Object-Type", "blob")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("X-HSP-Object-Type", kind)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

            self._json(404, {"error": "unknown"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b""
            path = self.path.split("?", 1)[0]  # e.g. /v1/sync/objects?scope=org

            if path == "/v1/sync/objects":
                return self._handle_put_objects(raw)

            if path.startswith("/v1/sync/refs/"):
                return self._handle_cas(raw)

            self._json(404, {"error": "unknown"})

        def _handle_put_objects(self, raw):
            # multipart/form-data: parse parts (field=hash, filename=type,
            # body=raw bytes). The server recomputes each hash and 422s on
            # mismatch (contract §4.2).
            ctype = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in ctype:
                return self._json(400, {"error": "expected multipart"})
            boundary = ctype.split("boundary=", 1)[1].encode("ascii")
            accepted, already = [], []
            parts = raw.split(b"--" + boundary)
            for part in parts:
                # Only trim the delimiter framing: a leading CRLF and a
                # trailing CRLF. Do NOT strip() the whole part -- that would
                # also eat legitimate trailing newlines from the object bytes.
                if part.startswith(b"\r\n"):
                    part = part[2:]
                if part.endswith(b"\r\n"):
                    part = part[:-2]
                if not part or part == b"--":
                    continue
                if b"\r\n\r\n" not in part:
                    continue
                headers_blob, body = part.split(b"\r\n\r\n", 1)
                hdr_text = headers_blob.decode("utf-8", "replace")
                claimed_hash = None
                kind = None
                for line in hdr_text.split("\r\n"):
                    if line.lower().startswith("content-disposition"):
                        for token in line.split(";"):
                            token = token.strip()
                            if token.startswith('name="'):
                                claimed_hash = token[len('name="'):-1]
                            elif token.startswith('filename="'):
                                kind = token[len('filename="'):-1]
                if claimed_hash is None:
                    continue
                real = "sha256:" + hashlib.sha256(body).hexdigest()
                if real != claimed_hash:
                    return self._json(422, {
                        "error": "hash_mismatch", "claimed": claimed_hash,
                    })
                if claimed_hash in state.objects:
                    already.append(claimed_hash)
                else:
                    state.objects[claimed_hash] = (kind, body)
                    accepted.append(claimed_hash)
            return self._json(200, {"accepted": accepted, "already_present": already})

        def _handle_cas(self, raw):
            from urllib.parse import unquote
            name = unquote(self.path[len("/v1/sync/refs/"):])
            body = json.loads(raw.decode("utf-8")) if raw else {}
            frm = body.get("from")
            to = body.get("to")
            # M2 (contract §11.5): a non-admin member's CAS on an org HEAD is
            # accept-always converted to a proposal → 202.
            if name.startswith("refs/org/") and not state.org_role_admin:
                n = len(state.proposals) + 1
                state.proposals.append({"n": n, "to": to, "base": frm})
                org = name.split("/")[2]
                prop_ref = f"refs/org/{org}/proposals/{n}"
                state.refs[prop_ref] = to
                return self._json(202, {"proposal_id": n, "ref": prop_ref})
            if state.force_conflict_once:
                state.force_conflict_once = False
                return self._json(409, {"actual": state.refs.get(name, "")})
            current = state.refs.get(name)
            if current != frm:
                return self._json(409, {"actual": current or ""})
            state.refs[name] = to
            return self._json(200, {"ref": name, "hash": to})

    return Handler


@pytest.fixture
def mock_server():
    state = _MockState()
    server = HTTPServer(("127.0.0.1", 0), _make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield base, state
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_skill(skills_dir: Path, name: str, body: str = "# skill\n", *, category=None):
    """Create a minimal skill dir under skills_dir; return its path."""
    parent = skills_dir / category if category else skills_dir
    d = parent / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: test\n---\n{body}", encoding="utf-8"
    )
    return d


def _jwt(claims: dict) -> str:
    import jwt as _pyjwt
    return _pyjwt.encode(claims, "x" * 32, algorithm="HS256")


# ---------------------------------------------------------------------------
# Content addressing & canonicalization (contract §2.1, §2.5, OI-5)
# ---------------------------------------------------------------------------

class TestAddressing:
    def test_full_64_hex_address(self):
        addr = ssc.hsp_address(b"")
        # sha256 of empty is the well-known e3b0... digest, full 64 hex.
        assert addr == (
            "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )
        assert len(addr.split(":", 1)[1]) == 64

    def test_address_differs_from_local_truncated_namespace(self):
        # OI-5: HSP full-64-hex must NOT equal the local truncated 16-hex form.
        data = b"hello world"
        full = ssc.hsp_address(data)
        truncated = "sha256:" + hashlib.sha256(data).hexdigest()[:16]
        assert full != truncated
        assert len(full.split(":")[1]) == 64
        assert len(truncated.split(":")[1]) == 16

    def test_canonical_json_sorted_no_whitespace(self):
        out = ssc.canonical_json_bytes({"b": 1, "a": 2})
        assert out == b'{"a":2,"b":1}'
        assert b" " not in out
        assert not out.endswith(b"\n")

    def test_canonical_json_stable(self):
        obj = {"type": "tree", "entries": [{"name": "x", "hash": "sha256:aa"}]}
        assert ssc.canonical_json_bytes(obj) == ssc.canonical_json_bytes(dict(obj))


# ---------------------------------------------------------------------------
# DEV-PHASE gate (tool_gateway_admin) + M1-D opt-in
# ---------------------------------------------------------------------------

class TestDevGate:
    def test_gate_open_with_claim(self, monkeypatch):
        token = _jwt({"sub": "user1", "tool_gateway_admin": True})
        monkeypatch.setattr(
            ssc, "resolve_nous_runtime_credentials",
            lambda **kw: {"api_key": token, "base_url": "https://x"}, raising=False,
        )
        # patch the lazily-imported symbol used inside resolve_identity
        import hermes_cli.auth as auth_mod
        monkeypatch.setattr(auth_mod, "resolve_nous_runtime_credentials",
                            lambda **kw: {"api_key": token, "base_url": "https://x"})
        ident = ssc.resolve_identity()
        assert ident["dev_gate_ok"] is True
        assert ident["owner"] == "user1"

    def test_gate_closed_without_claim(self, monkeypatch):
        token = _jwt({"sub": "user1"})  # no tool_gateway_admin
        import hermes_cli.auth as auth_mod
        monkeypatch.setattr(auth_mod, "resolve_nous_runtime_credentials",
                            lambda **kw: {"api_key": token, "base_url": "https://x"})
        ident = ssc.resolve_identity()
        assert ident["dev_gate_ok"] is False

    def test_gate_closed_when_claim_false(self, monkeypatch):
        token = _jwt({"sub": "u", "tool_gateway_admin": False})
        import hermes_cli.auth as auth_mod
        monkeypatch.setattr(auth_mod, "resolve_nous_runtime_credentials",
                            lambda **kw: {"api_key": token, "base_url": "https://x"})
        assert ssc.dev_gate_open() is False

    def test_maybe_push_inert_when_gate_closed(self, monkeypatch):
        token = _jwt({"sub": "u"})
        import hermes_cli.auth as auth_mod
        monkeypatch.setattr(auth_mod, "resolve_nous_runtime_credentials",
                            lambda **kw: {"api_key": token})
        monkeypatch.setattr(ssc, "resolve_sync_base_url", lambda: "http://x")
        # gate closed -> None (inert), never attempts a push
        assert ssc.maybe_push_skills() is None

    def test_maybe_pull_inert_when_not_logged_in(self, monkeypatch):
        import hermes_cli.auth as auth_mod

        def _raise(**kw):
            raise RuntimeError("not logged in")

        monkeypatch.setattr(auth_mod, "resolve_nous_runtime_credentials", _raise)
        assert ssc.maybe_pull_skills() is None


# ---------------------------------------------------------------------------
# Object building (contract §2.2-§2.4)
# ---------------------------------------------------------------------------

class TestObjectBuilding:
    def test_build_tree_blob_and_exec(self, tmp_path):
        d = tmp_path / "skill"
        d.mkdir()
        (d / "SKILL.md").write_text("hello", encoding="utf-8")
        script = d / "run.sh"
        script.write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
        script.chmod(0o755)

        objects = ssc.ObjectSet()
        tree_hash = ssc.build_tree(d, objects, max_object_bytes=ssc.DEFAULT_MAX_OBJECT_BYTES)
        assert tree_hash.startswith("sha256:")
        # tree object present and canonical
        kind, data = objects.objects[tree_hash]
        assert kind == ssc.KIND_TREE
        tree = json.loads(data)
        entries = {e["name"]: e for e in tree["entries"]}
        assert entries["SKILL.md"]["mode"] == ssc.MODE_FILE
        assert entries["run.sh"]["mode"] == ssc.MODE_EXEC
        # entries sorted by name (byte order)
        names = [e["name"] for e in tree["entries"]]
        assert names == sorted(names)

    def test_build_tree_dedups_identical_blobs(self, tmp_path):
        d = tmp_path / "skill"
        (d / "a").mkdir(parents=True)
        (d / "b").mkdir(parents=True)
        (d / "a" / "f.txt").write_text("same", encoding="utf-8")
        (d / "b" / "f.txt").write_text("same", encoding="utf-8")
        objects = ssc.ObjectSet()
        ssc.build_tree(d, objects, max_object_bytes=ssc.DEFAULT_MAX_OBJECT_BYTES)
        blob_hashes = [h for h, (k, _) in objects.objects.items() if k == ssc.KIND_BLOB]
        # only one unique blob for the identical "same" content
        assert len(set(blob_hashes)) == 1

    def test_build_tree_skips_symlink(self, tmp_path):
        d = tmp_path / "skill"
        d.mkdir()
        (d / "real.txt").write_text("x", encoding="utf-8")
        try:
            (d / "link.txt").symlink_to(d / "real.txt")
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unsupported here")
        objects = ssc.ObjectSet()
        tree_hash = ssc.build_tree(d, objects, max_object_bytes=ssc.DEFAULT_MAX_OBJECT_BYTES)
        tree = json.loads(objects.objects[tree_hash][1])
        names = [e["name"] for e in tree["entries"]]
        assert "link.txt" not in names
        assert "real.txt" in names

    def test_build_tree_rejects_oversize_blob(self, tmp_path):
        d = tmp_path / "skill"
        d.mkdir()
        (d / "big").write_bytes(b"x" * 100)
        objects = ssc.ObjectSet()
        with pytest.raises(ValueError):
            ssc.build_tree(d, objects, max_object_bytes=10)

    def test_build_commit_shape(self):
        objects = ssc.ObjectSet()
        c = ssc.build_commit(
            "sha256:tree", ["sha256:p"], owner="o", device="dev",
            message="m", objects=objects, ts="2026-07-18T00:00:00Z",
        )
        commit = json.loads(objects.objects[c][1])
        assert commit["type"] == "commit"
        assert commit["tree"] == "sha256:tree"
        assert commit["parents"] == ["sha256:p"]
        assert commit["author"] == {"owner": "o", "device": "dev"}
        assert commit["artifact_type"] == "skill"


# ---------------------------------------------------------------------------
# Three-way merge decision (contract §4.4, M1-C; mirrors skills_sync.py:619)
# ---------------------------------------------------------------------------

class TestMergeDecision:
    def test_no_change(self):
        assert ssc._merge_skill("b", "b", "b") == "either"

    def test_ours_only_changed(self):
        assert ssc._merge_skill("b", "o", "b") == "ours"

    def test_theirs_only_changed(self):
        assert ssc._merge_skill("b", "b", "t") == "theirs"

    def test_both_converged(self):
        assert ssc._merge_skill("b", "x", "x") == "either"

    def test_true_overlap(self):
        assert ssc._merge_skill("b", "o", "t") == "overlap"

    def test_deleted_both(self):
        assert ssc._merge_skill(None, None, None) == "none"


# ---------------------------------------------------------------------------
# End-to-end push / pull / conflict against the mock server
# ---------------------------------------------------------------------------

@pytest.fixture
def synced_env(tmp_path, monkeypatch):
    """A HERMES_HOME with two opted-in skills + a token-carrying identity."""
    import hermes_constants
    home = tmp_path / "hermes"
    skills = home / "skills"
    skills.mkdir(parents=True)
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: home)
    monkeypatch.setattr(ssc, "_skills_dir", lambda: skills)

    _write_skill(skills, "alpha", body="alpha v1\n")
    _write_skill(skills, "beta", body="beta v1\n", category="devops")

    # Opt both into sync + treat them as eligible (bypass bundled/hub checks).
    monkeypatch.setattr(ssc, "list_synced_skill_names", lambda: ["alpha", "beta"])

    def _rel(name):
        from pathlib import PurePosixPath
        return {"alpha": PurePosixPath("alpha"),
                "beta": PurePosixPath("devops/beta")}.get(name)

    monkeypatch.setattr(ssc, "_skill_rel_path", _rel)

    def _find(name):
        return {"alpha": skills / "alpha",
                "beta": skills / "devops" / "beta"}.get(name)

    import tools.skill_usage as su
    monkeypatch.setattr(su, "_find_skill_dir", _find)

    token = _jwt({"sub": "owner1", "tool_gateway_admin": True})
    identity = {"api_key": token, "base_url": "http://x", "owner": "owner1",
                "dev_gate_ok": True, "claims": {}}
    return home, skills, identity


class TestEndToEnd:
    def test_capabilities_version_check(self, mock_server):
        base, state = mock_server
        client = ssc.HSPClient(base, "tok")
        caps = client.capabilities()
        assert caps["hsp_version"] == "1"
        ssc._check_version(caps)  # no raise

    def test_version_mismatch_raises(self, mock_server):
        base, state = mock_server
        state.hsp_version = "2"
        client = ssc.HSPClient(base, "tok")
        with pytest.raises(ssc.HSPError):
            ssc._check_version(client.capabilities())

    def test_push_uploads_and_cas(self, mock_server, synced_env):
        base, state = mock_server
        home, skills, identity = synced_env
        client = ssc.HSPClient(base, identity["api_key"])
        result = ssc.push_skills(client, identity=identity)
        assert result["ok"] is True
        # HEAD ref advanced to our commit
        head = state.refs["refs/user/owner1/HEAD"]
        assert head == result["head"]
        # commit object is present and well-formed
        kind, data = state.objects[head]
        assert kind == ssc.KIND_COMMIT
        commit = json.loads(data)
        assert commit["author"]["owner"] == "owner1"
        assert commit["parents"] == []  # first commit

    def test_push_then_pull_materializes(self, mock_server, synced_env, tmp_path, monkeypatch):
        base, state = mock_server
        home, skills, identity = synced_env
        client = ssc.HSPClient(base, identity["api_key"])
        ssc.push_skills(client, identity=identity)

        # Simulate a fresh device: new skills dir, same server, same opt-in.
        dev2 = tmp_path / "hermes2" / "skills"
        dev2.mkdir(parents=True)
        monkeypatch.setattr(ssc, "_skills_dir", lambda: dev2)
        monkeypatch.setattr(ssc, "read_sync_state", lambda: {"head": None, "skills": {}})
        saved = {}
        monkeypatch.setattr(ssc, "write_sync_state", lambda d: saved.update(d))

        result = ssc.pull_skills(client, identity=identity)
        assert result["ok"] is True
        assert "alpha" in result["updated"]
        assert "devops/beta" in result["updated"]
        # content materialized to disk
        assert (dev2 / "alpha" / "SKILL.md").read_text().endswith("alpha v1\n")
        assert (dev2 / "devops" / "beta" / "SKILL.md").read_text().endswith("beta v1\n")

    def test_push_idempotent_reupload(self, mock_server, synced_env):
        base, state = mock_server
        home, skills, identity = synced_env
        client = ssc.HSPClient(base, identity["api_key"])
        r1 = ssc.push_skills(client, identity=identity)
        n_objects = len(state.objects)
        # push again with no local change -> same head, objects already_present
        r2 = ssc.push_skills(client, identity=identity)
        assert r2["ok"] is True
        assert r2["head"] == r1["head"]
        assert len(state.objects) == n_objects  # nothing new stored

    def test_conflict_nonoverlap_merges(self, mock_server, synced_env, monkeypatch):
        base, state = mock_server
        home, skills, identity = synced_env
        client = ssc.HSPClient(base, identity["api_key"])
        # First push establishes a base head we record locally.
        first = ssc.push_skills(client, identity=identity)
        # Inject a divergent server head: change beta server-side so the next
        # CAS loses. We simulate by forcing one 409 whose actual == current head
        # (the server keeps the same tree, so no overlap on alpha which we edit).
        (skills / "alpha" / "SKILL.md").write_text(
            "---\nname: alpha\ndescription: test\n---\nalpha v2\n", encoding="utf-8"
        )
        state.force_conflict_once = True
        result = ssc.push_skills(client, identity=identity)
        # actual == our own head -> both-sides identical -> merge commit succeeds
        assert result.get("ok") is True
        assert result.get("merged") is True

    def test_conflict_true_overlap_writes_conflict_ref(self, mock_server, synced_env, monkeypatch):
        base, state = mock_server
        home, skills, identity = synced_env
        client = ssc.HSPClient(base, identity["api_key"])
        ssc.push_skills(client, identity=identity)

        # Build a DIFFERENT server-side head for the SAME skill (alpha) so the
        # three-way merge sees a true overlap. We construct it via a second
        # snapshot after editing alpha differently, push it directly, then make
        # our local head stale and edit alpha a third way.
        (skills / "alpha" / "SKILL.md").write_text(
            "---\nname: alpha\ndescription: test\n---\nSERVER edit\n", encoding="utf-8"
        )
        objs, root, _ = ssc.snapshot_profile(["alpha", "beta"])
        their_commit = ssc.build_commit(
            root, [], owner="owner1", device="other", message="theirs", objects=objs
        )
        client.put_objects(objs.objects)
        state.refs["refs/user/owner1/HEAD"] = their_commit

        # Our local edit to the same skill, from the OLD base -> true overlap.
        (skills / "alpha" / "SKILL.md").write_text(
            "---\nname: alpha\ndescription: test\n---\nLOCAL edit\n", encoding="utf-8"
        )
        result = ssc.push_skills(client, identity=identity)
        assert result.get("conflict") is True
        assert result["conflict_ref"].startswith("refs/user/owner1/conflict/")
        assert "alpha" in result["overlapping_skills"]
        # a conflict ref head was written server-side
        assert result["conflict_ref"] in state.refs


# ---------------------------------------------------------------------------
# M1-D opt-in sidecar flag (tools/skill_usage.set_sync / is_sync_enabled)
# ---------------------------------------------------------------------------

class TestOptInFlag:
    def test_set_and_read_sync_flag(self, tmp_path, monkeypatch):
        import tools.skill_usage as su
        monkeypatch.setattr(su, "_skills_dir", lambda: tmp_path)
        # Make the skill curation-eligible so the gated mutator writes.
        monkeypatch.setattr(su, "is_curation_eligible", lambda name, *a, **k: True)

        assert su.is_sync_enabled("foo") is False
        su.set_sync("foo", True)
        assert su.is_sync_enabled("foo") is True
        su.set_sync("foo", False)
        assert su.is_sync_enabled("foo") is False

    def test_sync_flag_ignored_for_ineligible(self, tmp_path, monkeypatch):
        import tools.skill_usage as su
        monkeypatch.setattr(su, "_skills_dir", lambda: tmp_path)
        # Bundled/hub/external skills are not curation-eligible -> mutator no-ops.
        monkeypatch.setattr(su, "is_curation_eligible", lambda name, *a, **k: False)
        su.set_sync("bundled-skill", True)
        assert su.is_sync_enabled("bundled-skill") is False


# ---------------------------------------------------------------------------
# §2.8 sync-manifest — opt-in as content in the sync plane (cross-device)
# ---------------------------------------------------------------------------

class TestSyncManifest:
    def test_build_parse_roundtrip(self):
        data = ssc.build_sync_manifest_bytes({"beta": True, "alpha": False})
        parsed = ssc.parse_sync_manifest(data)
        assert parsed == {"alpha": False, "beta": True}

    def test_manifest_wire_shape(self):
        # Must match gateway-gateway src/sync/manifest.ts: type + version:1 +
        # skills:[{name,enabled}]. Skills sorted by name for a stable address.
        import json
        data = ssc.build_sync_manifest_bytes({"z": True, "a": True})
        obj = json.loads(data.decode("utf-8"))
        assert obj["type"] == "sync-manifest"
        assert obj["version"] == 1
        assert obj["skills"] == [
            {"name": "a", "enabled": True},
            {"name": "z", "enabled": True},
        ]

    def test_parse_rejects_malformed(self):
        # Strict: unknown type, bad version, non-array skills, malformed entry.
        assert ssc.parse_sync_manifest(b"not json") is None
        assert ssc.parse_sync_manifest(b'{"type":"nope","version":1,"skills":[]}') is None
        assert ssc.parse_sync_manifest(b'{"type":"sync-manifest","version":2,"skills":[]}') is None
        assert ssc.parse_sync_manifest(b'{"type":"sync-manifest","version":1,"skills":{}}') is None
        assert (
            ssc.parse_sync_manifest(
                b'{"type":"sync-manifest","version":1,"skills":[{"name":"x"}]}'
            )
            is None
        )
        # A malformed manifest must NOT be mistaken for "no skills opted in".
        assert ssc.parse_sync_manifest(b'{"type":"sync-manifest","version":1,"skills":[]}') == {}

    def test_snapshot_embeds_manifest_root_blob(self, mock_server, synced_env):
        # snapshot_profile must add a root-level `sync-manifest` blob recording
        # the opted-in set, alongside the skill subtrees, so opt-in is durable
        # plane content. Read it back via read_manifest_of_root.
        base, state = mock_server
        home, skills, identity = synced_env
        client = ssc.HSPClient(base, identity["api_key"])

        objs, root_hash, skill_map = ssc.snapshot_profile(["alpha", "beta"])
        client.put_objects(objs.objects)

        manifest = ssc.read_manifest_of_root(client, root_hash)
        assert manifest == {"alpha": True, "beta": True}

        # The manifest is a root-level BLOB, not a skill subtree, so the skill
        # walk must not surface it as a skill.
        trees = ssc._skill_trees_of_root(client, root_hash)
        assert "sync-manifest" not in trees
        assert set(trees) == {"alpha", "devops/beta"}

    def test_pull_adopts_opt_in_from_manifest(self, mock_server, synced_env, monkeypatch):
        # A skill opted in on device A (present + enabled in the plane manifest)
        # becomes opted in locally on pull, even if this device had it disabled.
        base, state = mock_server
        home, skills, identity = synced_env
        client = ssc.HSPClient(base, identity["api_key"])

        # Device A pushes alpha+beta (manifest enables both).
        ssc.push_skills(client, identity=identity)

        # Simulate device B: local opt-in intent is EMPTY, but eligibility passes.
        adopted = {}
        import tools.skill_usage as su
        monkeypatch.setattr(su, "is_curation_eligible", lambda name, *a, **k: True)
        monkeypatch.setattr(su, "is_sync_enabled", lambda name: False)
        monkeypatch.setattr(su, "set_sync", lambda name, val: adopted.__setitem__(name, val))
        # Local head unknown so the pull actually runs.
        monkeypatch.setattr(ssc, "read_sync_state", lambda: {"head": None, "skills": {}})
        monkeypatch.setattr(ssc, "write_sync_state", lambda d: None)
        # No local opt-in gate (so materialize isn't the thing under test).
        monkeypatch.setattr(ssc, "_opted_in_rel_paths", lambda: [])

        result = ssc.pull_skills(client, identity=identity)
        assert result["ok"] is True
        # Both skills from the plane manifest were adopted into local opt-in.
        assert adopted == {"alpha": True, "beta": True}
        assert set(result["opt_in_adopted"]) == {"alpha", "beta"}


# ---------------------------------------------------------------------------
# Env-var configuration (Hermes Cloud "on by default" via environment)
# ---------------------------------------------------------------------------

class TestEnvConfig:
    def test_base_url_env_wins(self, monkeypatch):
        monkeypatch.setenv("HERMES_SYNC_BASE_URL", "https://plane.example/")
        assert ssc.resolve_sync_base_url() == "https://plane.example"

    def test_feature_enabled_env(self, monkeypatch):
        # Default off.
        monkeypatch.delenv("HERMES_SYNC_ENABLED", raising=False)
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {}, raising=False)
        assert ssc.sync_feature_enabled() is False
        for truthy in ("1", "true", "YES", "on"):
            monkeypatch.setenv("HERMES_SYNC_ENABLED", truthy)
            assert ssc.sync_feature_enabled() is True
        for falsy in ("0", "false", "off"):
            monkeypatch.setenv("HERMES_SYNC_ENABLED", falsy)
            assert ssc.sync_feature_enabled() is False

    def test_default_opt_in_env(self, monkeypatch):
        monkeypatch.delenv("HERMES_SYNC_DEFAULT_OPT_IN", raising=False)
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {}, raising=False)
        assert ssc.sync_default_opt_in() is False
        monkeypatch.setenv("HERMES_SYNC_DEFAULT_OPT_IN", "true")
        assert ssc.sync_default_opt_in() is True

    def test_config_yaml_fallback_when_no_env(self, monkeypatch):
        monkeypatch.delenv("HERMES_SYNC_ENABLED", raising=False)
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"sync": {"enabled": True}},
            raising=False,
        )
        assert ssc.sync_feature_enabled() is True

    def test_env_overrides_config_yaml(self, monkeypatch):
        # Env wins over config.yaml (operator override precedence).
        monkeypatch.setenv("HERMES_SYNC_ENABLED", "false")
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"sync": {"enabled": True}},
            raising=False,
        )
        assert ssc.sync_feature_enabled() is False

    def test_opt_out_policy_syncs_all_eligible(self, monkeypatch):
        # With opt-out on, every eligible skill syncs even with no `sync:true`
        # flag; an explicit `sync:false` still excludes.
        monkeypatch.setattr(ssc, "sync_default_opt_in", lambda: True)
        monkeypatch.setattr(ssc, "_all_local_skill_names", lambda: ["alpha", "beta", "gamma"])
        monkeypatch.setattr(ssc, "is_sync_eligible", lambda n: n in {"alpha", "beta", "gamma"})
        import tools.skill_usage as su
        # gamma explicitly opted out; alpha/beta have no flag.
        monkeypatch.setattr(su, "load_usage", lambda: {"gamma": {"sync": False}})
        assert ssc.list_synced_skill_names() == ["alpha", "beta"]

    def test_opt_in_policy_requires_flag(self, monkeypatch):
        # With opt-out OFF (default opt-in), only explicitly-enabled skills sync.
        monkeypatch.setattr(ssc, "sync_default_opt_in", lambda: False)
        monkeypatch.setattr(ssc, "is_sync_eligible", lambda n: True)
        import tools.skill_usage as su
        monkeypatch.setattr(
            su, "load_usage",
            lambda: {"alpha": {"sync": True}, "beta": {}, "gamma": {"sync": False}},
        )
        assert ssc.list_synced_skill_names() == ["alpha"]


class TestDeviceName:
    def test_default_is_hostname_seeded(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ssc, "_skills_dir", lambda: tmp_path)
        monkeypatch.delenv("HERMES_SYNC_DEVICE_NAME", raising=False)
        monkeypatch.setattr(
            "socket.gethostname", lambda: "bens-macbook.local", raising=False
        )
        val = ssc.stable_device_id()
        # short hostname + short suffix, NOT a bare 32-char hash
        assert val.startswith("bens-macbook-")
        assert val != "bens-macbook-"
        # persisted + stable across calls
        assert (tmp_path / ".sync_device_id").read_text() == val
        assert ssc.stable_device_id() == val

    def test_existing_file_wins_over_default_and_env(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ssc, "_skills_dir", lambda: tmp_path)
        (tmp_path / ".sync_device_id").write_text("Explicit Name", encoding="utf-8")
        monkeypatch.setenv("HERMES_SYNC_DEVICE_NAME", "cloud-seed")
        assert ssc.stable_device_id() == "Explicit Name"

    def test_env_seeds_first_use(self, tmp_path, monkeypatch):
        # Hermes Cloud path: HERMES_SYNC_DEVICE_NAME seeds the first-use label.
        monkeypatch.setattr(ssc, "_skills_dir", lambda: tmp_path)
        monkeypatch.setenv("HERMES_SYNC_DEVICE_NAME", "hermes-cloud-ben-1")
        assert ssc.stable_device_id() == "hermes-cloud-ben-1"
        # persisted so it stays stable even if the env later changes
        assert (tmp_path / ".sync_device_id").read_text() == "hermes-cloud-ben-1"
        monkeypatch.setenv("HERMES_SYNC_DEVICE_NAME", "changed")
        assert ssc.stable_device_id() == "hermes-cloud-ben-1"

    def test_set_device_name_overwrites(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ssc, "_skills_dir", lambda: tmp_path)
        (tmp_path / ".sync_device_id").write_text("old", encoding="utf-8")
        stored = ssc.set_device_name("  Ben's Laptop  ")
        assert stored == "Ben's Laptop"  # trimmed
        assert ssc.stable_device_id() == "Ben's Laptop"

    def test_set_device_name_rejects_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ssc, "_skills_dir", lambda: tmp_path)
        import pytest

        with pytest.raises(ValueError):
            ssc.set_device_name("   ")


# ---------------------------------------------------------------------------
# M2 org-shared skills (contract §11): identity gate, pull, propose (202/merge)
# ---------------------------------------------------------------------------

def _org_identity(role=None, org_id="org-1", owner="owner1"):
    claims = {"sub": owner, "org_id": org_id, "tool_gateway_admin": True}
    if role is not None:
        claims["org_role"] = role
    token = _jwt(claims)
    return {"api_key": token, "base_url": "http://x", "owner": owner,
            "dev_gate_ok": True, "claims": claims,
            **({"org_id": org_id, "org_role": role} if role else {})}


class TestOrgIdentityGate:
    def test_org_identity_requires_role_claim(self, monkeypatch):
        # Personal org: NAS stamps NO org_role -> inert, not an error path.
        token = _jwt({"sub": "u", "org_id": "org-1"})
        import hermes_cli.auth as auth_mod
        monkeypatch.setattr(auth_mod, "resolve_nous_runtime_credentials",
                            lambda **kw: {"api_key": token, "base_url": "https://x"})
        with pytest.raises(ssc.SyncInertError):
            ssc.resolve_org_identity()
        assert ssc.org_sync_available() is False

    def test_org_identity_with_role(self, monkeypatch):
        token = _jwt({"sub": "u", "org_id": "org-9", "org_role": "MEMBER"})
        import hermes_cli.auth as auth_mod
        monkeypatch.setattr(auth_mod, "resolve_nous_runtime_credentials",
                            lambda **kw: {"api_key": token, "base_url": "https://x"})
        ident = ssc.resolve_org_identity()
        assert ident["org_id"] == "org-9"
        assert ident["org_role"] == "MEMBER"
        assert ssc.org_sync_available() is True

    def test_org_mirror_excluded_from_personal_sync(self, tmp_path, monkeypatch):
        # A skill under _org/<id>/ must never be personal-sync eligible.
        skills = tmp_path / "skills"
        org_skill = skills / "_org" / "org-1" / "shared-x"
        org_skill.mkdir(parents=True)
        (org_skill / "SKILL.md").write_text("---\nname: shared-x\n---\n")
        monkeypatch.setattr(ssc, "_skills_dir", lambda: skills)
        import tools.skill_usage as su
        monkeypatch.setattr(su, "is_bundled", lambda n: False)
        monkeypatch.setattr(su, "is_hub_installed", lambda n: False)
        monkeypatch.setattr(su, "_find_skill_dir", lambda n: org_skill)
        import agent.skill_utils as sku
        monkeypatch.setattr(sku, "is_external_skill_path", lambda p: False)
        assert ssc.is_sync_eligible("shared-x") is False


class TestOrgEndToEnd:
    def test_admin_propose_merges_directly(self, mock_server, synced_env):
        base, state = mock_server
        home, skills, identity = synced_env
        identity = {**identity, "org_id": "org-1", "org_role": "ADMIN"}
        client = ssc.HSPClient(base, identity["api_key"])
        result = ssc.propose_skill("alpha", client, identity=identity)
        assert result["ok"] is True
        assert result.get("merged") is True
        head = state.refs["refs/org/org-1/HEAD"]
        assert head == result["head"]
        commit = json.loads(state.objects[head][1])
        assert commit["parents"] == []  # first org commit

    def test_member_propose_becomes_202_proposal(self, mock_server, synced_env):
        base, state = mock_server
        home, skills, identity = synced_env
        # Seed an org HEAD as admin first.
        admin_ident = {**identity, "org_id": "org-1", "org_role": "ADMIN"}
        client = ssc.HSPClient(base, identity["api_key"])
        seeded = ssc.propose_skill("alpha", client, identity=admin_ident)

        # Member edits beta and proposes: server converts to 202.
        state.org_role_admin = False
        (skills / "devops" / "beta" / "SKILL.md").write_text(
            "---\nname: beta\n---\nbeta v2 member edit\n", encoding="utf-8"
        )
        member_ident = {**identity, "org_id": "org-1", "org_role": "MEMBER"}
        result = ssc.propose_skill("beta", client, identity=member_ident)
        assert result["ok"] is True
        assert result.get("proposal_pending") is True
        assert result["proposal_id"] == 1
        # HEAD untouched; proposal ref parked at the member's commit.
        assert state.refs["refs/org/org-1/HEAD"] == seeded["head"]
        assert state.refs["refs/org/org-1/proposals/1"] == result["commit"]
        # NEVER reported as merged.
        assert "merged" not in result

    def test_member_proposal_splices_not_replaces(self, mock_server, synced_env):
        # The proposed root must keep the OTHER skills from HEAD (per-skill
        # delta, not a wholesale replace).
        base, state = mock_server
        home, skills, identity = synced_env
        admin_ident = {**identity, "org_id": "org-1", "org_role": "ADMIN"}
        client = ssc.HSPClient(base, identity["api_key"])
        ssc.propose_skill("alpha", client, identity=admin_ident)
        ssc.propose_skill("beta", client, identity=admin_ident)

        state.org_role_admin = False
        member_ident = {**identity, "org_id": "org-1", "org_role": "MEMBER"}
        result = ssc.propose_skill("alpha", client, identity=member_ident)
        # Walk the proposed commit's root: both skills present.
        commit = json.loads(state.objects[result["commit"]][1])
        root = json.loads(state.objects[commit["tree"]][1])
        names = {e["name"] for e in root["entries"]}
        assert "alpha" in names and "devops" in names

    def test_pull_org_skills_materializes_mirror(self, mock_server, synced_env):
        base, state = mock_server
        home, skills, identity = synced_env
        admin_ident = {**identity, "org_id": "org-1", "org_role": "ADMIN"}
        client = ssc.HSPClient(base, identity["api_key"])
        ssc.propose_skill("alpha", client, identity=admin_ident)

        result = ssc.pull_org_skills(client, identity=admin_ident)
        assert result["ok"] is True
        assert "alpha" in result["updated"]
        mirrored = skills / "_org" / "org-1" / "alpha" / "SKILL.md"
        assert mirrored.exists()
        assert mirrored.read_text().endswith("alpha v1\n")

    def test_pull_org_noop_when_no_head(self, mock_server, synced_env):
        base, state = mock_server
        home, skills, identity = synced_env
        ident = {**identity, "org_id": "org-1", "org_role": "MEMBER"}
        client = ssc.HSPClient(base, identity["api_key"])
        result = ssc.pull_org_skills(client, identity=ident)
        assert result["ok"] is True
        assert result["head"] is None
        assert result["updated"] == []

    def test_propose_requires_org_feature(self, mock_server, synced_env):
        base, state = mock_server
        home, skills, identity = synced_env
        state.org_feature = False
        ident = {**identity, "org_id": "org-1", "org_role": "ADMIN"}
        client = ssc.HSPClient(base, identity["api_key"])
        with pytest.raises(ssc.SyncInertError):
            ssc.propose_skill("alpha", client, identity=ident)

    def test_maybe_pull_org_inert_without_role(self, monkeypatch):
        # Personal org: no org_role claim -> None, never raises.
        token = _jwt({"sub": "u", "org_id": "org-1"})
        import hermes_cli.auth as auth_mod
        monkeypatch.setattr(auth_mod, "resolve_nous_runtime_credentials",
                            lambda **kw: {"api_key": token})
        assert ssc.maybe_pull_org_skills() is None
