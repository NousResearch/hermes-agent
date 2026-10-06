"""Capture target selection must never implicitly target Cua Driver itself."""

from tools.computer_use.cua_backend_capture import _select_capture_target


def _window(app_name: str, pid: int, window_id: int, z_index: int):
    return {
        "app_name": app_name,
        "pid": pid,
        "window_id": window_id,
        "title": "",
        "off_screen": False,
        "z_index": z_index,
    }


def test_implicit_capture_skips_frontmost_cua_driver_window():
    driver = _window("Cua Driver", 55543, 101, 6)
    app = _window("IB Gateway 10.50", 777, 202, 5)

    target = _select_capture_target(
        [driver, app], app_requested=False, exact_target=False
    )

    assert target == app


def test_all_self_windows_keep_existing_failure_target():
    frontmost = _window("Cua Driver", 55543, 101, 6)
    other = _window("cua-driver", 55543, 102, 5)

    target = _select_capture_target(
        [frontmost, other], app_requested=False, exact_target=False
    )

    assert target == frontmost


def test_exact_capture_keeps_explicit_cua_driver_target():
    driver = _window("Cua_Driver", 55543, 101, 6)
    app = _window("IB Gateway 10.50", 777, 202, 5)

    target = _select_capture_target(
        [driver, app], app_requested=False, exact_target=True
    )

    assert target == driver


def test_similarly_named_user_app_is_not_treated_as_driver_self():
    docs = _window("Cua Driver Docs", 888, 303, 7)
    app = _window("IB Gateway 10.50", 777, 202, 5)

    target = _select_capture_target(
        [docs, app], app_requested=False, exact_target=False
    )

    assert target == docs
