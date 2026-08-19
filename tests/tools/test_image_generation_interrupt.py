"""_wait_fal_result must notice a user interrupt while the FAL job runs."""

import threading
import time

import pytest

import tools.image_generation_tool as image_tool
from tools.interrupt import set_interrupt


class _SlowHandler:
    """Fake FAL handler whose get() blocks like the real SDK."""

    def __init__(self, delay=30.0, result=None):
        self.delay = delay
        self._result = result if result is not None else {"images": []}

    def get(self):
        time.sleep(self.delay)
        return self._result


class _FastHandler:
    def __init__(self, result):
        self._result = result

    def get(self):
        return self._result


@pytest.fixture(autouse=True)
def _clean_interrupt():
    set_interrupt(False)
    yield
    set_interrupt(False)


def test_wait_fal_result_returns_result():
    result = image_tool._wait_fal_result(_FastHandler({"images": [{"url": "u"}]}))
    assert result == {"images": [{"url": "u"}]}


def test_wait_fal_result_raises_on_interrupt():
    def _interrupt_soon(tid):
        time.sleep(0.2)
        set_interrupt(True, tid)

    tid = threading.current_thread().ident
    threading.Thread(target=_interrupt_soon, args=(tid,), daemon=True).start()

    t0 = time.monotonic()
    with pytest.raises(image_tool.ImageGenerationInterrupted):
        image_tool._wait_fal_result(_SlowHandler(delay=30.0), poll_seconds=0.05)
    assert time.monotonic() - t0 < 5.0


def test_wait_fal_result_propagates_handler_error():
    class _ErrHandler:
        def get(self):
            raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        image_tool._wait_fal_result(_ErrHandler())


def test_upscale_interrupt_propagates(monkeypatch):
    """_upscale_image must NOT swallow the interrupt into a None fallback."""

    monkeypatch.setattr(
        image_tool, "_submit_fal_request", lambda *a, **k: _SlowHandler(30.0)
    )

    def _interrupt_soon(tid):
        time.sleep(0.2)
        set_interrupt(True, tid)

    tid = threading.current_thread().ident
    threading.Thread(target=_interrupt_soon, args=(tid,), daemon=True).start()

    with pytest.raises(image_tool.ImageGenerationInterrupted):
        image_tool._upscale_image("https://example.com/x.png", "prompt")
