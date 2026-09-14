import json
import shutil
import tempfile
import pytest

from workstation.vault import VaultManager
import tools.vault_tools as vt


@pytest.fixture
def mock_vault_manager(monkeypatch):
    temp_dir = tempfile.mkdtemp()
    mgr = VaultManager(vault_dir=temp_dir)
    monkeypatch.setattr(vt, "_vault_manager", mgr)
    yield mgr
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_vault_tools_handlers(mock_vault_manager):
    # 1. Write Note
    write_res_raw = vt._handle_vault_write({
        "title": "Quantum Computing",
        "content": "Quantum Computing uses [[Qubits]] and principles of [[Superposition]].",
        "tags": ["physics", "computing"]
    })
    write_res = json.loads(write_res_raw)
    assert write_res["status"] == "ok"
    assert write_res["note"]["title"] == "Quantum Computing"

    # 2. Read Note
    read_res_raw = vt._handle_vault_read({"title": "Quantum Computing"})
    read_res = json.loads(read_res_raw)
    assert read_res["status"] == "ok"
    assert "physics" in read_res["note"]["tags"]
    assert "Qubits" in read_res["note"]["forward_links"]

    # 3. Write Qubits Note with backlink
    vt._handle_vault_write({
        "title": "Qubits",
        "content": "A qubit is the basic unit of quantum info.",
        "tags": ["quantum"]
    })

    # 4. Check Backlinks
    backlinks_raw = vt._handle_vault_backlinks({"title": "Qubits"})
    backlinks_res = json.loads(backlinks_raw)
    assert backlinks_res["status"] == "ok"
    assert "Quantum Computing" in backlinks_res["backlinks"]

    # 5. Append
    append_res_raw = vt._handle_vault_append({
        "title": "Qubits",
        "content": "Also related to [[Entanglement]]."
    })
    append_res = json.loads(append_res_raw)
    assert append_res["status"] == "ok"

    # 6. Search
    search_res_raw = vt._handle_vault_search({"query": "physics"})
    search_res = json.loads(search_res_raw)
    assert search_res["status"] == "ok"
    assert len(search_res["results"]) >= 1

    # 7. Graph
    graph_res_raw = vt._handle_vault_graph({})
    graph_res = json.loads(graph_res_raw)
    assert graph_res["status"] == "ok"
    assert graph_res["nodes_count"] >= 2
