"""Content, not file metadata, is the authority for cached OAuth pool ownership."""

import json
import os

import pytest


@pytest.mark.platforms("any")
@pytest.mark.asyncio
@pytest.mark.parametrize("shared", [False, True])
async def test_endpoint_edit_revokes_authority_even_when_metadata_collides(tmp_path, monkeypatch, shared):
    import httpx

    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.mcp_oauth import OAuthStorageDetachedError, PoolAuthority
    from tools.mcp_oauth_manager import MCPOAuthManager

    # Model a filesystem with no change-time signal, without pretending the host is Windows.
    # Real in-place I/O below preserves size, inode and mtime on every host.
    monkeypatch.setattr("utils.file_signature", lambda st: (st.st_mtime_ns, st.st_size, st.st_ino, 0))
    root = tmp_path / "root"
    home = root / "profiles" / "worker" if shared else root
    home.mkdir(parents=True)
    original = ("mcp_servers:\n  team:\n    auth: oauth\n"
                "    url: https://a.example/mcp\n    oauth: {share_with_profiles: true}\n")
    (root / "config.yaml").write_text(original, encoding="utf-8")
    if shared:
        (home / "config.yaml").write_text(original, encoding="utf-8")
    pool = root / "mcp-tokens" / "team.json"
    authority = PoolAuthority.capture(str(pool), home, "team", None)
    assert authority.revocation() is None
    pool.parent.mkdir()
    pool.write_text(json.dumps({"access_token": "SYNTHETIC_GRANT", "token_type": "Bearer"}))
    scope = set_hermes_home_override(home)
    try:
        provider = MCPOAuthManager().get_or_build_provider("team", "https://a.example/mcp", None)
    finally:
        reset_hermes_home_override(scope)
    assert provider is not None
    # Seed the SDK's already-authorized state; no discovery or provider traffic is needed.
    provider.context.current_tokens = await provider.context.storage.get_tokens()
    provider._initialized = True
    first = httpx.Request("GET", "https://a.example/mcp")
    flow = provider.async_auth_flow(first)
    try:
        assert await anext(flow) is first
        assert first.headers["Authorization"] == "Bearer SYNTHETIC_GRANT"
    finally:
        await flow.aclose()

    config = root / "config.yaml"
    before = config.stat()
    with config.open("r+", encoding="utf-8") as stream:
        stream.write(original.replace("a.example", "b.example"))
    os.utime(config, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = config.stat()
    assert (before.st_size, before.st_ino, before.st_mtime_ns) == (
        after.st_size, after.st_ino, after.st_mtime_ns)
    revoked = authority.revocation()
    assert revoked is not None and revoked[1], "the old endpoint must lose authority"
    request = httpx.Request("GET", "https://a.example/mcp")
    flow = provider.async_auth_flow(request)
    try:
        with pytest.raises(OAuthStorageDetachedError):
            await anext(flow)
        assert "Authorization" not in request.headers
    finally:
        await flow.aclose()


@pytest.mark.platforms("any")
def test_authority_reads_fresh_documents_and_never_hides_read_errors(tmp_path, monkeypatch):
    import tools.mcp_oauth as oauth
    import utils

    config = tmp_path / "config.yaml"
    first = b"\xef\xbb\xbfmcp_servers: {team: {url: 'https://a.example/mcp'}}\n"
    second = first.replace(b"a.example", b"b.example")
    config.write_bytes(first)
    real_parse = utils.fast_safe_load
    parsed = []

    def parse_then_replace(document):
        parsed.append(document)
        data = real_parse(document)
        config.write_bytes(second)
        return data

    monkeypatch.setattr(utils, "fast_safe_load", parse_then_replace)
    assert oauth._load_mcp_server_config(tmp_path, "team")["url"] == "https://a.example/mcp"
    assert oauth._load_mcp_server_config(tmp_path, "team")["url"] == "https://b.example/mcp"
    assert oauth._load_mcp_server_config(tmp_path, "team")["url"] == "https://b.example/mcp"
    assert len(parsed) == 3, "authority must re-read rather than trust metadata"

    from hermes_cli import config_backend
    real_read = config_backend.read_config_doc

    def deny_config_read(path):
        if path == config:
            raise PermissionError("synthetic unreadable config")
        return real_read(path)

    with monkeypatch.context() as patch:
        patch.setattr(config_backend, "read_config_doc", deny_config_read)
        assert oauth._read_config_as_written(tmp_path) is oauth._CONFIG_UNREADABLE
    config.write_bytes(b"\xff")
    assert oauth._read_config_as_written(tmp_path) is oauth._CONFIG_UNREADABLE
    config.unlink()
    assert oauth._read_config_as_written(tmp_path) is None
    config.write_bytes(second)
    assert oauth._load_mcp_server_config(tmp_path, "team")["url"] == "https://b.example/mcp"


def test_authority_uses_selected_backend_without_a_local_file(tmp_path, monkeypatch):
    from hermes_cli import config_backend
    from tools.mcp_oauth import _load_mcp_server_config

    class Backend:
        url = "https://a.example/mcp"

        def read_user_doc(self, home):
            assert home == tmp_path
            return {"mcp_servers": {"team": {"url": self.url}}}

    backend = Backend()
    monkeypatch.setattr(config_backend, "get_config_backend", lambda: backend)
    assert _load_mcp_server_config(tmp_path, "team")["url"] == "https://a.example/mcp"
    backend.url = "https://b.example/mcp"
    assert _load_mcp_server_config(tmp_path, "team")["url"] == "https://b.example/mcp"
    assert not (tmp_path / "config.yaml").exists()
