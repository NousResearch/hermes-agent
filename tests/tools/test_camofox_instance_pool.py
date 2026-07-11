from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.camofox_instance_pool import CamofoxInstancePool


def test_same_task_reuses_live_instance(tmp_path):
    (tmp_path / "server.js").write_text("// test")
    process = MagicMock()
    process.poll.return_value = None
    pool = CamofoxInstancePool(tmp_path)

    with patch("tools.camofox_instance_pool.subprocess.Popen", return_value=process) as popen, patch.object(
        pool, "_wait_until_ready"
    ):
        first = pool.get_or_start("thread-a")
        second = pool.get_or_start("thread-a")

    assert first is second
    popen.assert_called_once()


def test_concurrent_same_task_waits_for_one_ready_instance(tmp_path):
    (tmp_path / "server.js").write_text("// test")
    process = MagicMock()
    process.poll.return_value = None
    pool = CamofoxInstancePool(tmp_path)

    with patch("tools.camofox_instance_pool.subprocess.Popen", return_value=process) as popen, patch.object(
        pool, "_wait_until_ready"
    ):
        with ThreadPoolExecutor(max_workers=2) as executor:
            first, second = executor.map(pool.get_or_start, ["thread-a", "thread-a"])

    assert first is second
    popen.assert_called_once()


def test_different_tasks_get_distinct_port_triples(tmp_path):
    pool = CamofoxInstancePool(tmp_path)
    with patch.object(pool, "_port_available", return_value=True):
        first = set(pool._ports_for_task("thread-a"))
        second = set(pool._ports_for_task("thread-b"))

    assert first.isdisjoint(second)


def test_spawn_sets_independent_api_and_vnc_ports(tmp_path):
    (tmp_path / "server.js").write_text("// test")
    process = MagicMock()
    process.poll.return_value = None
    pool = CamofoxInstancePool(tmp_path)

    with patch.object(pool, "_ports_for_task", return_value=(19401, 19402, 19403)), patch(
        "tools.camofox_instance_pool.subprocess.Popen", return_value=process
    ) as popen, patch.object(pool, "_wait_until_ready"):
        instance = pool.get_or_start("thread-a")

    env = popen.call_args.kwargs["env"]
    assert env["CAMOFOX_PORT"] == "19401"
    assert env["VNC_PORT"] == "19402"
    assert env["NOVNC_PORT"] == "19403"
    assert env["VNC_BIND"] == "127.0.0.1"
    assert instance.api_url == "http://127.0.0.1:19401"
    assert instance.viewer_url.startswith("http://127.0.0.1:19403/vnc.html")


def test_start_failure_terminates_process(tmp_path):
    (tmp_path / "server.js").write_text("// test")
    process = MagicMock()
    process.poll.return_value = None
    pool = CamofoxInstancePool(tmp_path)

    with patch("tools.camofox_instance_pool.subprocess.Popen", return_value=process), patch.object(
        pool, "_wait_until_ready", side_effect=RuntimeError("boom")
    ), patch("tools.camofox_instance_pool.os.getpgid", return_value=4321), patch(
        "tools.camofox_instance_pool.os.killpg"
    ) as killpg:
        with pytest.raises(RuntimeError, match="boom"):
            pool.get_or_start("thread-a")

    killpg.assert_called_once_with(4321, 15)


def test_dead_server_reaps_viewer_and_closes_log_before_replacement(tmp_path):
    (tmp_path / "server.js").write_text("// test")
    dead = MagicMock(pid=100)
    dead.poll.return_value = 1
    viewer = MagicMock(pid=101)
    viewer.poll.return_value = None
    log_file = MagicMock()
    pool = CamofoxInstancePool(tmp_path)
    from tools.camofox_instance_pool import CamofoxInstance
    pool._instances["thread-a"] = CamofoxInstance(
        "thread-a", 19401, 19402, 19403, dead,
        viewer_process=viewer, log_file=log_file,
    )
    replacement = MagicMock()
    replacement.poll.return_value = None

    with patch("tools.camofox_instance_pool.subprocess.Popen", return_value=replacement), patch.object(
        pool, "_wait_until_ready"
    ), patch("tools.camofox_instance_pool.os.getpgid", return_value=777), patch(
        "tools.camofox_instance_pool.os.killpg"
    ) as killpg:
        pool.get_or_start("thread-a")

    killpg.assert_called_once_with(777, 15)
    log_file.close.assert_called_once()


def test_popup_viewer_uses_dedicated_profile_and_instance_url(tmp_path):
    server_dir = tmp_path / "server"
    server_dir.mkdir()
    (server_dir / "server.js").write_text("// test")
    executable = tmp_path / "camoufox"
    executable.write_text("")
    server_process = MagicMock()
    server_process.poll.return_value = None
    viewer_process = MagicMock()
    viewer_process.poll.return_value = None
    pool = CamofoxInstancePool(
        server_dir,
        launch_viewer=True,
        viewer_executable=executable,
        viewer_profile_root=tmp_path / "profiles",
    )

    with patch.object(pool, "_ports_for_task", return_value=(19401, 19402, 19403)), patch(
        "tools.camofox_instance_pool.subprocess.Popen",
        side_effect=[server_process, viewer_process],
    ) as popen, patch.object(pool, "_wait_until_ready"), patch(
        "tools.camofox_instance_pool.requests.get"
    ) as get:
        get.return_value.status_code = 200
        instance = pool.get_or_start("thread-a")
        pool.ensure_viewer(instance)

    viewer_args = popen.call_args_list[1].args[0]
    assert viewer_args[0] == str(executable)
    assert "--new-instance" in viewer_args
    assert "--profile" in viewer_args
    assert instance.viewer_url == viewer_args[-1]
    assert instance.viewer_process is viewer_process


def test_missing_server_fails_before_spawn(tmp_path):
    pool = CamofoxInstancePool(Path(tmp_path))
    with patch("tools.camofox_instance_pool.subprocess.Popen") as popen:
        with pytest.raises(RuntimeError, match="server.js not found"):
            pool.get_or_start("thread-a")
    popen.assert_not_called()
