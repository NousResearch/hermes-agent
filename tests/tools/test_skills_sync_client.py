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
                return self._json(200, {
                    "hsp_version": state.hsp_version,
                    "features": ["personal"],
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

            if self.path == "/v1/sync/objects":
                return self._handle_put_objects(raw)

            if self.path.startswith("/v1/sync/refs/"):
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
        monkeypatch.setattr(ssc, "read_sync_manifest", lambda: {"head": None, "skills": {}})
        saved = {}
        monkeypatch.setattr(ssc, "write_sync_manifest", lambda d: saved.update(d))

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
        monkeypatch.setattr(ssc, "read_sync_manifest", lambda: {"head": None, "skills": {}})
        monkeypatch.setattr(ssc, "write_sync_manifest", lambda d: None)
        # No local opt-in gate (so materialize isn't the thing under test).
        monkeypatch.setattr(ssc, "_opted_in_rel_paths", lambda: [])

        result = ssc.pull_skills(client, identity=identity)
        assert result["ok"] is True
        # Both skills from the plane manifest were adopted into local opt-in.
        assert adopted == {"alpha": True, "beta": True}
        assert set(result["opt_in_adopted"]) == {"alpha", "beta"}
