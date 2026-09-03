import errno

from tools.thread_exhaustion import is_thread_start_exhaustion


def test_runtime_thread_start_message_matches():
    assert is_thread_start_exhaustion(RuntimeError("can't start new thread"))


def test_resource_unavailable_os_error_matches():
    assert is_thread_start_exhaustion(
        OSError(errno.EAGAIN, "Resource temporarily unavailable")
    )


def test_unrelated_errors_do_not_match():
    assert not is_thread_start_exhaustion(RuntimeError("boom"))
    assert not is_thread_start_exhaustion(OSError(errno.EAGAIN, "database busy"))
    assert not is_thread_start_exhaustion(
        RuntimeError("cannot schedule new futures after interpreter shutdown")
    )
