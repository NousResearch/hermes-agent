"""Wake-word voice mode tests."""

import json
import sys
import types
from pathlib import Path


def _frame(value: int = 0, samples: int = 1280):
    import numpy as np

    return (np.ones((samples, 1), dtype=np.int16) * value)


class _FakeOpenWakeWordModel:
    def __init__(self, scores):
        self._scores = list(scores)
        self.calls = 0

    def predict(self, frame, **kwargs):
        self.calls += 1
        score = self._scores.pop(0) if self._scores else 0.0
        return {"hermes": score}


def _install_fake_openwakeword(monkeypatch, tmp_path: Path, *, model_cls=None):
    pkg_dir = tmp_path / "openwakeword"
    pkg_dir.mkdir(exist_ok=True)
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")

    fake_openwakeword = types.SimpleNamespace(
        __file__=str(pkg_dir / "__init__.py"),
        MODELS={
            "hey_jarvis": {
                "model_path": str(pkg_dir / "resources" / "models" / "hey_jarvis_v0.1.tflite"),
                "download_url": "https://example.test/hey_jarvis_v0.1.tflite",
            }
        },
        FEATURE_MODELS={
            "melspectrogram": {
                "model_path": str(pkg_dir / "resources" / "models" / "melspectrogram.tflite"),
                "download_url": "https://example.test/melspectrogram.tflite",
            },
            "embedding": {
                "model_path": str(pkg_dir / "resources" / "models" / "embedding_model.tflite"),
                "download_url": "https://example.test/embedding_model.tflite",
            },
        },
    )
    monkeypatch.setitem(sys.modules, "openwakeword", fake_openwakeword)
    if model_cls is not None:
        monkeypatch.setitem(sys.modules, "openwakeword.model", types.SimpleNamespace(Model=model_cls))
    return fake_openwakeword


def test_load_wake_config_defaults():
    from tools.wake_word import FRAME_SAMPLES, WAKE_MIN_SILENCE_DURATION_SECONDS, load_wake_config

    cfg = load_wake_config({"voice": {}})

    assert cfg.provider == "openwakeword"
    assert cfg.threshold == 0.5
    assert cfg.patience_frames == 2
    assert cfg.vad_threshold == 0.25
    assert cfg.pre_roll_ms == 1200
    assert cfg.dialog_timeout_seconds == 45.0
    assert cfg.max_utterance_seconds == 30.0
    assert cfg.silence_duration == WAKE_MIN_SILENCE_DURATION_SECONDS
    assert cfg.frame_samples == FRAME_SAMPLES
    assert cfg.training.positive_samples == 50
    assert cfg.training.negative_samples == 30
    assert cfg.training.ambient_seconds == 60
    assert cfg.training.command == ""


def test_load_wake_config_reads_training_command():
    from tools.wake_word import load_wake_config

    cfg = load_wake_config(
        {
            "voice": {
                "wake": {
                    "training": {
                        "command": "python train_wake.py",
                    },
                },
            },
        }
    )

    assert cfg.training.command == "python train_wake.py"


def test_load_wake_config_clamps_short_silence_duration():
    from tools.wake_word import WAKE_MIN_SILENCE_DURATION_SECONDS, load_wake_config

    cfg = load_wake_config({"voice": {"wake": {"silence_duration": 1.2}}})

    assert cfg.silence_duration == WAKE_MIN_SILENCE_DURATION_SECONDS


def test_load_wake_config_uses_voice_threshold_for_wake_recording():
    from tools.wake_word import load_wake_config

    cfg = load_wake_config(
        {
            "voice": {
                "silence_threshold": 200,
                "wake": {"silence_threshold": 700},
            }
        }
    )

    assert cfg.silence_threshold == 200


def test_detector_triggers_only_after_patience_frames():
    from tools.wake_word import OpenWakeWordDetector, load_wake_config

    fake_model = _FakeOpenWakeWordModel([0.7, 0.8])
    cfg = load_wake_config(
        {
            "voice": {
                "wake": {
                    "model_path": "/tmp/hermes.onnx",
                    "threshold": 0.5,
                    "patience_frames": 2,
                }
            }
        }
    )
    detector = OpenWakeWordDetector(cfg, model_factory=lambda _cfg: fake_model)

    assert detector.process_frame(_frame()) is False
    assert detector.last_score == 0.7
    assert detector.process_frame(_frame()) is True
    assert detector.last_score == 0.8


def test_default_model_factory_uses_sibling_onnx_assets(tmp_path: Path, monkeypatch):
    from tools import wake_word

    model_path = tmp_path / "hey_jarvis_v0.1.onnx"
    melspec_path = tmp_path / "melspectrogram.onnx"
    embedding_path = tmp_path / "embedding_model.onnx"
    model_path.write_bytes(b"wake")
    melspec_path.write_bytes(b"melspec")
    embedding_path.write_bytes(b"embedding")

    pkg_dir = tmp_path / "openwakeword"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    fake_openwakeword = types.SimpleNamespace(__file__=str(pkg_dir / "__init__.py"))
    captured: dict = {}

    class FakeModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "openwakeword", fake_openwakeword)
    monkeypatch.setitem(sys.modules, "openwakeword.model", types.SimpleNamespace(Model=FakeModel))

    cfg = wake_word.load_wake_config(
        {"voice": {"wake": {"model_path": str(model_path), "vad_threshold": 0.25}}}
    )
    wake_word._default_model_factory(cfg)

    assert captured["wakeword_models"] == [str(model_path)]
    assert captured["inference_framework"] == "onnx"
    assert captured["melspec_model_path"] == str(melspec_path)
    assert captured["embedding_model_path"] == str(embedding_path)
    assert captured["vad_threshold"] == 0.0


def test_resolve_wake_model_path_uses_cached_pretrained_for_stale_temp_path(tmp_path: Path, monkeypatch):
    from tools import wake_word

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _install_fake_openwakeword(monkeypatch, tmp_path)
    cache_dir = tmp_path / ".hermes" / "wake_words" / "openwakeword" / "v0.5.1" / "onnx"
    cache_dir.mkdir(parents=True)
    cached_model = cache_dir / "hey_jarvis_v0.1.onnx"
    cached_model.write_bytes(b"wake")
    (cache_dir / "melspectrogram.onnx").write_bytes(b"melspec")
    (cache_dir / "embedding_model.onnx").write_bytes(b"embedding")

    cfg = wake_word.load_wake_config(
        {
            "voice": {
                "wake": {
                    "phrase": "hey jarvis",
                    "model_path": "/tmp/hermes_wake_models/hey_jarvis_v0.1.onnx",
                }
            }
        }
    )

    assert wake_word.resolve_wake_model_path(cfg) == cached_model


def test_check_wake_requirements_downloads_known_pretrained_model(tmp_path: Path, monkeypatch):
    from tools import wake_word

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _install_fake_openwakeword(monkeypatch, tmp_path)
    monkeypatch.setattr(wake_word, "_audio_available", lambda: True)
    monkeypatch.setattr(wake_word, "_openwakeword_model_available", lambda: True)

    downloaded: list[str] = []

    def fake_download(url: str, target_path: Path) -> None:
        downloaded.append(url)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(b"asset")

    monkeypatch.setattr(wake_word, "_download_file", fake_download)
    cfg = wake_word.load_wake_config(
        {
            "voice": {
                "wake": {
                    "phrase": "hey jarvis",
                    "model_path": "/tmp/hermes_wake_models/hey_jarvis_v0.1.onnx",
                }
            }
        }
    )

    result = wake_word.check_wake_requirements(cfg, download_missing=True)

    assert result["available"] is True
    assert result["model_available"] is True
    assert result["resolved_model_path"].endswith("/hey_jarvis_v0.1.onnx")
    assert downloaded == [
        "https://example.test/hey_jarvis_v0.1.onnx",
        "https://example.test/melspectrogram.onnx",
        "https://example.test/embedding_model.onnx",
    ]


def test_default_model_factory_uses_cached_feature_assets_for_custom_onnx(tmp_path: Path, monkeypatch):
    from tools import wake_word

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    model_path = tmp_path / "custom.onnx"
    model_path.write_bytes(b"wake")
    cache_dir = tmp_path / ".hermes" / "wake_words" / "openwakeword" / "v0.5.1" / "onnx"
    cache_dir.mkdir(parents=True)
    melspec_path = cache_dir / "melspectrogram.onnx"
    embedding_path = cache_dir / "embedding_model.onnx"
    melspec_path.write_bytes(b"melspec")
    embedding_path.write_bytes(b"embedding")

    captured: dict = {}

    class FakeModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    _install_fake_openwakeword(monkeypatch, tmp_path, model_cls=FakeModel)
    cfg = wake_word.load_wake_config({"voice": {"wake": {"model_path": str(model_path)}}})

    wake_word._default_model_factory(cfg)

    assert captured["wakeword_models"] == [str(model_path)]
    assert captured["inference_framework"] == "onnx"
    assert captured["melspec_model_path"] == str(melspec_path)
    assert captured["embedding_model_path"] == str(embedding_path)


def test_listener_keeps_pre_roll_when_wake_detects():
    from tools.wake_word import WakeWordListener, load_wake_config

    fake_model = _FakeOpenWakeWordModel([0.0, 0.0, 0.9])
    cfg = load_wake_config(
        {
            "voice": {
                "wake": {
                    "model_path": "/tmp/hermes.onnx",
                    "threshold": 0.5,
                    "patience_frames": 1,
                    "pre_roll_ms": 240,
                }
            }
        }
    )
    listener = WakeWordListener(
        cfg,
        on_transcript=lambda _text: None,
        model_factory=lambda _cfg: fake_model,
        async_transcribe=False,
    )

    listener.process_frame(_frame(10))
    listener.process_frame(_frame(20))
    listener.process_frame(_frame(300))

    assert listener.state == "recording"
    assert [int(frame[0][0]) for frame in listener.recording_frames] == [10, 20, 300]


def test_listener_dialog_timeout_returns_to_passive():
    from tools.wake_word import WakeWordListener, load_wake_config

    now = [100.0]
    cfg = load_wake_config(
        {"voice": {"wake": {"model_path": "/tmp/hermes.onnx", "dialog_timeout_seconds": 45}}}
    )
    listener = WakeWordListener(
        cfg,
        on_transcript=lambda _text: None,
        model_factory=lambda _cfg: _FakeOpenWakeWordModel([0.0]),
        time_fn=lambda: now[0],
    )

    listener.enter_dialog()
    assert listener.state == "dialog"
    now[0] += 46.0
    listener.process_frame(_frame())

    assert listener.state == "passive"


def test_listener_pause_prevents_detection_until_resume():
    from tools.wake_word import WakeWordListener, load_wake_config

    fake_model = _FakeOpenWakeWordModel([0.95])
    cfg = load_wake_config(
        {"voice": {"wake": {"model_path": "/tmp/hermes.onnx", "threshold": 0.5}}}
    )
    listener = WakeWordListener(
        cfg,
        on_transcript=lambda _text: None,
        model_factory=lambda _cfg: fake_model,
    )

    listener.pause("tts")
    listener.process_frame(_frame(400))
    assert fake_model.calls == 0
    assert listener.state == "paused"

    listener.resume()
    listener.process_frame(_frame(400))
    assert fake_model.calls == 1


def test_listener_keeps_recording_through_short_pause():
    from tools.wake_word import WakeWordListener, load_wake_config

    now = [100.0]
    cfg = load_wake_config(
        {
            "voice": {
                "wake": {
                    "model_path": "/tmp/hermes.onnx",
                    "silence_duration": 1.2,
                    "silence_threshold": 100,
                }
            }
        }
    )
    listener = WakeWordListener(
        cfg,
        on_transcript=lambda _text: None,
        model_factory=lambda _cfg: _FakeOpenWakeWordModel([0.0]),
        time_fn=lambda: now[0],
    )

    with listener._lock:
        listener._recording_started_at = now[0]
        listener._last_speech_at = now[0]
        listener._has_speech = True

    now[0] += 1.3
    assert listener._finish_reason_locked() == ""

    now[0] += 1.6
    assert listener._finish_reason_locked() == ""

    now[0] += 0.2
    assert listener._finish_reason_locked() == "silence"


def test_strip_wake_phrase_from_transcript():
    from tools.wake_word import strip_wake_phrase

    assert strip_wake_phrase("Гермес, сколько времени?", "Гермес") == "сколько времени?"
    assert strip_wake_phrase("гермес сделай заметку", "Гермес") == "сделай заметку"
    assert strip_wake_phrase("Без ключевого слова", "Гермес") == "Без ключевого слова"


def test_trainer_writes_model_metadata_and_config_update(tmp_path: Path):
    from tools.wake_word import WakeWordTrainer, load_wake_config

    def collect_samples(config, dataset_dir):
        positive = []
        negative = []
        ambient = []
        for bucket, count, target in (
            ("positive", config.training.positive_samples, positive),
            ("negative", config.training.negative_samples, negative),
            ("ambient", 1, ambient),
        ):
            bucket_dir = dataset_dir / bucket
            bucket_dir.mkdir(parents=True, exist_ok=True)
            for idx in range(count):
                sample_path = bucket_dir / f"{idx}.wav"
                sample_path.write_bytes(b"RIFFsample")
                target.append(sample_path)
        return {"positive": positive, "negative": negative, "ambient": ambient}

    def train_backend(config, dataset_dir, output_path):
        output_path.write_bytes(b"onnx-model")

    cfg = load_wake_config(
        {"voice": {"wake": {"phrase": "Гермес", "training": {"positive_samples": 2, "negative_samples": 1}}}}
    )
    trainer = WakeWordTrainer(
        cfg,
        output_root=tmp_path,
        sample_collector=collect_samples,
        training_backend=train_backend,
    )

    result = trainer.train()

    model_path = Path(result["model_path"])
    metadata_path = model_path.parent / "metadata.json"
    assert model_path.exists()
    assert json.loads(metadata_path.read_text())["phrase"] == "Гермес"
    assert result["config_update"]["voice.wake.model_path"] == str(model_path)
    assert result["counts"]["positive"] == 2
    assert result["counts"]["negative"] == 1


def test_trainer_command_backend_uses_configured_command(tmp_path: Path, monkeypatch):
    from tools import wake_word

    captured = {}

    def fake_run(command, *, env, cwd, **kwargs):
        captured.update({"command": command, "env": env, "cwd": cwd, "kwargs": kwargs})
        Path(env["HERMES_WAKE_OUTPUT_PATH"]).write_bytes(b"onnx-model")
        return types.SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(wake_word.subprocess, "run", fake_run)
    cfg = wake_word.load_wake_config(
        {"voice": {"wake": {"phrase": "Hermes", "training": {"command": "python train_wake.py"}}}}
    )
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    output_path = tmp_path / "model.onnx"

    wake_word.WakeWordTrainer(cfg)._run_training_command_backend(cfg, dataset_dir, output_path)

    assert captured["command"] == "python train_wake.py"
    assert captured["cwd"] == str(tmp_path)
    assert captured["env"]["HERMES_WAKE_PHRASE"] == "Hermes"
    assert captured["env"]["HERMES_WAKE_DATASET_DIR"] == str(dataset_dir)
    assert captured["env"]["HERMES_WAKE_OUTPUT_PATH"] == str(output_path)
