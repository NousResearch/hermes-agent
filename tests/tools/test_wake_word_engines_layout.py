"""Tests for tools.wake_word_engines — the sherpa KWS tokenizer-layout selection.

The engine must tokenize keywords against the layout a model directory actually ships:
the classic bpe layout (bpe.model beside tokens.txt) or the zh-en / wenetspeech phone
layout, whose ``*.phone`` lexicon needs ``phone+ppinyin``. Hard-coding "bpe" made every
phone-layout model — including the shipped zipformer-zh-en KWS model — crash on the
missing bpe.model (#110003).

No sherpa-onnx and no audio: the ``sherpa_onnx`` module is faked, the engine's
dependencies are stubbed, and the engine is built against a tmp model directory.
"""

import sys
import types
from pathlib import Path

import pytest

from tools import wake_word_engines as wwe


class _StubWw:
    """The slice of tools.wake_word that ``_SherpaKwsEngine._build()`` uses."""

    @staticmethod
    def _get(cfg, key, default=None):
        return cfg.get(key, default)

    @staticmethod
    def _active_profile_name():
        return "default"

    @staticmethod
    def enrolled_profile_phrases():
        return {}

    @staticmethod
    def _sensitivity(cfg):
        return 0.5


def _fake_sherpa(record: dict, *, text2token_error=None):
    mod = types.ModuleType("sherpa_onnx")

    def text2token(texts, **kwargs):
        record["texts"] = list(texts)
        record.update(kwargs)
        if text2token_error is not None:
            raise text2token_error
        return [[t] for t in texts]

    class _Spotter:
        def __init__(self, **kwargs):
            record["spotter"] = kwargs

        def create_stream(self):
            return types.SimpleNamespace()

    mod.text2token = text2token
    mod.KeywordSpotter = _Spotter
    return mod


def _model_dir(tmp_path: Path, layout: str) -> Path:
    d = tmp_path / "sherpa-kws-test"
    d.mkdir()
    (d / "tokens.txt").write_text("<blk> 0\n<unk> 1\n", encoding="utf-8")
    for part in ("encoder", "decoder", "joiner"):
        # _build() globs "{part}-*[!8].onnx": the int8 companion must not be picked.
        (d / f"{part}-epoch-1.onnx").write_bytes(b"\0")
        (d / f"{part}-epoch-1.int8.onnx").write_bytes(b"\0")
    if layout == "bpe":
        (d / "bpe.model").write_bytes(b"\0")
    elif layout == "phone":
        (d / "en.phone").write_text("a AA\n", encoding="utf-8")
    return d


@pytest.fixture
def build(monkeypatch, tmp_path):
    monkeypatch.setattr(wwe, "_ww", lambda: _StubWw)
    monkeypatch.setattr(wwe, "_ensure_dep", lambda feature, cfg: None)

    def _build(layout, *, text2token_error=None):
        record: dict = {}
        monkeypatch.setitem(sys.modules, "sherpa_onnx",
                            _fake_sherpa(record, text2token_error=text2token_error))
        model_dir = _model_dir(tmp_path, layout)
        wwe._SherpaKwsEngine({"phrase": "hey hermes", "profile_routing": False,
                              "sherpa": {"model_dir": str(model_dir)}})
        return record

    return _build


def test_phone_layout_uses_phone_ppinyin(build):
    record = build("phone")
    assert record["tokens_type"] == "phone+ppinyin"
    assert Path(record["lexicon"]).name == "en.phone"
    assert "bpe_model" not in record


def test_bpe_layout_still_uses_bpe(build):
    record = build("bpe")
    assert record["tokens_type"] == "bpe"
    assert Path(record["bpe_model"]).name == "bpe.model"
    assert "lexicon" not in record


def test_unknown_layout_raises_a_clear_error(build):
    with pytest.raises(RuntimeError, match=r"neither bpe\.model nor a \*\.phone lexicon"):
        build("neither")


def test_missing_pypinyin_names_the_dependency(build):
    with pytest.raises(RuntimeError, match="pypinyin"):
        build("phone", text2token_error=ModuleNotFoundError("No module named 'pypinyin'"))
