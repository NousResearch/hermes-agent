"""Tests for the ``stt.local.beam_size`` knob in build_local_transcribe_kwargs.

The default must stay 5. It is faster-whisper's accuracy-oriented setting and
was previously hardcoded, so anything that reads config without the key — batch
jobs, ``hermes transcribe`` — has to keep decoding exactly as it did before.
Only callers that opt in get the faster greedy decode.
"""

from tools.transcription_tools import build_local_transcribe_kwargs


def _beam(local_cfg):
    return build_local_transcribe_kwargs({"local": local_cfg})["beam_size"]


class TestBeamSizeDefault:
    def test_absent_key_keeps_the_accurate_default(self):
        assert _beam({}) == 5

    def test_absent_local_section_keeps_the_accurate_default(self):
        assert build_local_transcribe_kwargs({})["beam_size"] == 5


class TestBeamSizeOptIn:
    def test_greedy_is_honoured(self):
        assert _beam({"beam_size": 1}) == 1

    def test_wider_beam_is_honoured(self):
        assert _beam({"beam_size": 8}) == 8

    def test_numeric_string_is_accepted(self):
        """Config round-tripped through YAML/JSON can hand back a string."""
        assert _beam({"beam_size": "1"}) == 1


class TestBeamSizeRejectsNonsense:
    def test_below_one_clamps_to_greedy(self):
        """beam_size=0 would make faster-whisper raise; clamp instead of crash."""
        assert _beam({"beam_size": 0}) == 1
        assert _beam({"beam_size": -3}) == 1

    def test_unparseable_falls_back_to_the_default(self):
        assert _beam({"beam_size": "fast"}) == 5
        assert _beam({"beam_size": None}) == 5
        assert _beam({"beam_size": [1]}) == 5


class TestBeamSizeLeavesOtherKwargsAlone:
    def test_anti_hallucination_kwargs_are_still_present(self):
        """This helper is the single owner of the hardening kwargs; don't drop them."""
        base = build_local_transcribe_kwargs({"local": {}})
        opted = build_local_transcribe_kwargs({"local": {"beam_size": 1}})
        assert base.keys() == opted.keys()
        assert {k: v for k, v in base.items() if k != "beam_size"} == \
               {k: v for k, v in opted.items() if k != "beam_size"}
