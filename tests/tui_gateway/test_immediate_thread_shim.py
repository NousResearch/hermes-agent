"""The shared synchronous Thread stand-in must honor the ``threading.Thread`` contract (#137194).

On win32 ``subprocess.Popen._communicate`` builds its pipe-reader threads as
``Thread(target=self._readerthread, args=(fh, buffer))``, sets ``daemon``, calls
``start()``, then ``join(...)`` + ``is_alive()``. The historical inline
``_ImmediateThread`` shims dropped ``args`` (TypeError at ``start()``) and had no
``join``/``is_alive`` (AttributeError right after), so every win32 test that
captured subprocess output under the shim died while Linux — which reads pipes
with selectors — stayed green.
"""

from tests._fixtures.thread_shim import ImmediateThread


def test_start_forwards_args_and_kwargs_to_target():
    seen: list[tuple[int, str]] = []
    thread = ImmediateThread(
        target=lambda a, b=0: seen.append((a, b)),
        args=(1,),
        kwargs={"b": 2},
    )
    thread.start()
    assert seen == [(1, 2)]


def test_start_without_target_is_a_no_op():
    ImmediateThread().start()


def test_join_and_is_alive_present_finished_result():
    thread = ImmediateThread(target=lambda: None)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_daemon_and_unknown_thread_options_are_accepted():
    thread = ImmediateThread(target=lambda: None, daemon=True, name="worker")
    thread.daemon = True
    thread.start()
    assert not thread.is_alive()
