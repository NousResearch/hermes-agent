"""Tests for the computer_use toolset: capture payload shape and disk spill.

Split out of ``tests/tools/test_computer_use.py`` (issue #79991) as a
byte-for-byte move -- the capture-payload cluster: element label capping,
bounds-space notes and bounds scale, the ``elements_file`` spill, and bounded
screenshot persistence.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import patch

import pytest

# Shared fixtures stay in the original module; import them so these tests keep
# the exact same setup/teardown (``_reset_backend`` is autouse and rides along
# on the import).
from tests.tools.test_computer_use import _reset_backend  # noqa: F401


class TestCapturePayloadBudget:
    """Element labels and the aux-vision branch must respect response budgets.

    Regression tests for the Discord/Electron capture blowup: UIA exposes
    entire message bodies as element labels, so a single capture response
    exceeded 170KB and the model never saw the elements it needed.
    """

    def test_element_label_is_capped_in_json(self):
        from tools.computer_use.backend import UIElement
        from tools.computer_use.tool import _MAX_ELEMENT_LABEL_CHARS, _element_to_dict

        e = UIElement(index=3, role="Document", label="m" * 5000,
                      bounds=(0, 0, 100, 100), app="chrome.exe")
        d = _element_to_dict(e)
        assert len(d["label"]) == _MAX_ELEMENT_LABEL_CHARS
        assert d["label_truncated"] is True

    def test_short_label_not_flagged(self):
        from tools.computer_use.backend import UIElement
        from tools.computer_use.tool import _element_to_dict

        d = _element_to_dict(UIElement(index=0, role="Button", label="OK",
                                       bounds=(0, 0, 10, 10), app=""))
        assert d["label"] == "OK"
        assert "label_truncated" not in d

    def test_aux_vision_branch_respects_element_cap(self):
        """The aux-vision payload must carry the same capped element list as
        every other capture branch, not the full untruncated tree."""
        from tools.computer_use.backend import CaptureResult, UIElement
        from tools.computer_use import tool as cu_tool

        elements = [
            UIElement(index=i, role="Button", label=f"btn{i}",
                      bounds=(0, 0, 10, 10), app="")
            for i in range(50)
        ]
        cap = CaptureResult(mode="som", width=1024, height=768,
                            png_b64="iVBORw0KGgo=", elements=elements,
                            app="X", window_title="t", png_bytes_len=10)
        with patch("model_tools._run_async",
                   return_value=json.dumps({"analysis": "a screen"})):
            out = cu_tool._route_capture_through_aux_vision(
                cap, "summary",
                visible_elements=elements[:5], truncated_elements=45,
            )
        assert out is not None
        payload = json.loads(out)
        assert len(payload["elements"]) == 5
        assert payload["total_elements"] == 50
        assert payload["truncated_elements"] == 45


class TestBoundsSpaceNote:
    def test_note_present_when_bounds_exceed_image(self):
        from tools.computer_use.backend import UIElement
        from tools.computer_use.tool import _bounds_space_note

        # Live repro: 1455x791 screenshot, element bounds out to x=3840
        # (native 4K desktop space).
        elems = [UIElement(index=0, role="Button", label="Close",
                           bounds=(3771, 0, 69, 60), app="")]
        note = _bounds_space_note(elems, 1455, 791)
        assert note is not None
        assert "native desktop coordinates" in note

    def test_no_note_when_spaces_match(self):
        from tools.computer_use.backend import UIElement
        from tools.computer_use.tool import _bounds_space_note

        elems = [UIElement(index=0, role="Button", label="OK",
                           bounds=(10, 10, 50, 20), app="")]
        assert _bounds_space_note(elems, 1455, 791) is None

    def test_no_note_for_empty_or_degenerate(self):
        from tools.computer_use.backend import UIElement
        from tools.computer_use.tool import _bounds_space_note

        assert _bounds_space_note([], 1455, 791) is None
        zero = [UIElement(index=0, role="B", label="x",
                          bounds=(0, 0, 0, 0), app="")]
        assert _bounds_space_note(zero, 1455, 791) is None
        assert _bounds_space_note(zero, 0, 0) is None


class TestElementSpillFile:
    """Detail dropped from the in-context capture must be recoverable on disk."""

    def _dense_capture(self):
        from tools.computer_use.backend import CaptureResult, UIElement

        elems = [
            UIElement(index=i, role="Document", label=f"msg {i}: " + "x" * 2000,
                      bounds=(100 + i, 200, 3600, 60), app="chrome.exe")
            for i in range(120)
        ]
        return CaptureResult(mode="som", width=1455, height=791, png_b64=None,
                             elements=elems, app="chrome.exe",
                             window_title="Discord", png_bytes_len=0)

    def test_spill_file_holds_full_untruncated_tree(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.computer_use.tool import _capture_response

        out = json.loads(_capture_response(self._dense_capture()))
        assert "elements_file" in out
        assert str(out["elements_file"]) in out["summary"]
        spill = json.loads(
            open(out["elements_file"], encoding="utf-8").read())
        # Everything the in-context response dropped is in the file:
        assert spill["total_elements"] == 120
        assert len(spill["elements"]) == 120           # beyond max_elements cap
        assert len(spill["elements"][0]["label"]) > 2000  # beyond label cap
        assert spill["elements"][119]["label"].startswith("msg 119")

    def test_no_spill_when_nothing_dropped(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.computer_use.backend import CaptureResult, UIElement
        from tools.computer_use.tool import _capture_response

        cap = CaptureResult(mode="som", width=1455, height=791, png_b64=None,
                            elements=[UIElement(index=0, role="Button",
                                                label="OK",
                                                bounds=(10, 10, 50, 20),
                                                app="")],
                            app="X", window_title="t", png_bytes_len=0)
        out = json.loads(_capture_response(cap))
        assert "elements_file" not in out

    def test_spill_pruning_bounds_cache_growth(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.computer_use import tool as cu_tool

        cap = self._dense_capture()
        for _ in range(cu_tool._MAX_SPILL_FILES + 5):
            assert cu_tool._spill_elements_to_file(cap) is not None
        cache = tmp_path / "cache" / "computer_use"
        assert len(list(cache.glob("elements_*.json"))) <= cu_tool._MAX_SPILL_FILES

    def test_spill_failure_never_breaks_capture(self, monkeypatch):
        from tools.computer_use import tool as cu_tool

        monkeypatch.setattr(cu_tool, "_spill_elements_to_file",
                            lambda cap: None)
        out = json.loads(cu_tool._capture_response(self._dense_capture()))
        # Capture still succeeds and stays budget-capped without the file.
        assert out["truncated_elements"] == 20
        assert "elements_file" not in out


class TestCaptureScreenshotPersistence:
    """Image captures expose a bounded file for explicit user delivery."""

    _PNG_B64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76L"
        "AAAADUlEQVR4nGNgGAUgAAABCAABgukLHQAAAABJRU5ErkJggg=="
    )

    def _capture(self):
        from tools.computer_use.backend import CaptureResult

        return CaptureResult(
            mode="vision",
            width=8,
            height=8,
            png_b64=self._PNG_B64,
            image_mime_type="image/png",
            png_bytes_len=len(base64.b64decode(self._PNG_B64)),
        )

    def test_multimodal_capture_exposes_shareable_screenshot(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.computer_use import tool as cu_tool

        monkeypatch.setattr(
            cu_tool, "_should_route_through_aux_vision", lambda: False,
        )
        out = cu_tool._capture_response(self._capture())

        screenshot_path = out["meta"]["screenshot_path"]
        assert screenshot_path in out["text_summary"]
        assert "MEDIA:" not in out["text_summary"]
        assert screenshot_path.startswith(str(tmp_path / "cache" / "images"))
        assert Path(screenshot_path).read_bytes() == base64.b64decode(self._PNG_B64)

    def test_capture_cache_is_bounded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.computer_use import tool as cu_tool

        monkeypatch.setattr(cu_tool, "_MAX_CAPTURE_FILES", 2)
        for _ in range(3):
            assert cu_tool._persist_capture_image(self._capture()) is not None

        captures = list((tmp_path / "cache" / "images").glob("computer_use_*.*"))
        assert len(captures) == 2


class TestBoundsScaleField:
    def test_scale_reported_when_spaces_diverge(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.computer_use.backend import CaptureResult, UIElement
        from tools.computer_use.tool import _capture_response

        # Live repro geometry: 1455x791 screenshot, native bounds to 3799.
        elems = [UIElement(index=0, role="Button", label="Close",
                           bounds=(3730, 0, 69, 60), app="")]
        cap = CaptureResult(mode="som", width=1455, height=791, png_b64=None,
                            elements=elems, app="chrome.exe",
                            window_title="", png_bytes_len=0)
        out = json.loads(_capture_response(cap))
        assert out["bounds_scale"] == pytest.approx(3799 / 1455, abs=0.01)
        assert f"~{out['bounds_scale']}x" in out["summary"]

    def test_no_scale_when_spaces_match(self):
        from tools.computer_use.backend import UIElement
        from tools.computer_use.tool import _bounds_scale

        elems = [UIElement(index=0, role="Button", label="OK",
                           bounds=(10, 10, 50, 20), app="")]
        assert _bounds_scale(elems, 1455, 791) is None
        assert _bounds_scale([], 1455, 791) is None
        assert _bounds_scale(elems, 0, 0) is None
