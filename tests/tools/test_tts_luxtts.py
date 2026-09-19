"""Focused coverage for the optional LuxTTS local provider."""

import sys
from types import ModuleType

import pytest


@pytest.fixture(autouse=True)
def clear_luxtts_cache():
    from tools import tts_tool_local

    tts_tool_local._luxtts_model_cache.clear()
    yield
    tts_tool_local._luxtts_model_cache.clear()


@pytest.fixture
def fake_luxtts(monkeypatch):
    class Wave:
        def numpy(self):
            return self

        def squeeze(self):
            return [0.0, 0.1]

    class Model:
        instances = []

        def __init__(self, model, device):
            self.model, self.device = model, device
            self.encode_calls = []
            self.generate_calls = []
            Model.instances.append(self)

        def encode_prompt(self, path, duration, rms):
            self.encode_calls.append((path, duration, rms))
            return "encoded-prompt"

        def generate_speech(self, text, prompt, **kwargs):
            self.generate_calls.append((text, prompt, kwargs))
            return Wave()

    zipvoice = ModuleType("zipvoice")
    luxvoice = ModuleType("zipvoice.luxvoice")
    luxvoice.LuxTTS = Model
    soundfile = ModuleType("soundfile")
    soundfile.write = lambda path, audio, rate: open(path, "wb").write(b"RIFFfake")
    monkeypatch.setitem(sys.modules, "zipvoice", zipvoice)
    monkeypatch.setitem(sys.modules, "zipvoice.luxvoice", luxvoice)
    monkeypatch.setitem(sys.modules, "soundfile", soundfile)
    return Model


def test_luxtts_reuses_model_and_encoded_reference_prompt(tmp_path, fake_luxtts):
    from tools.tts_tool import _generate_luxtts

    reference = tmp_path / "consenting-speaker.wav"
    reference.write_bytes(b"RIFFfake")
    config = {"luxtts": {"ref_audio": str(reference), "device": "cpu", "num_steps": 5,
                          "t_shift": 0.8, "speed": 1.2, "ref_duration": 4}}

    _generate_luxtts("First sentence.", str(tmp_path / "first.wav"), config)
    _generate_luxtts("Second sentence.", str(tmp_path / "second.wav"), config)

    assert len(fake_luxtts.instances) == 1
    model = fake_luxtts.instances[0]
    assert model.encode_calls == [(str(reference), 4, 0.01)]
    assert model.generate_calls[0][2] == {"num_steps": 5, "t_shift": 0.8, "speed": 1.2, "return_smooth": False}


def test_luxtts_requested_cuda_falls_back_to_cpu_without_cuda(monkeypatch):
    from tools.tts_tool_local import _resolve_luxtts_device

    class Cuda:
        @staticmethod
        def is_available():
            return False

    torch = ModuleType("torch")
    torch.cuda = Cuda()
    torch.backends = type("Backends", (), {"mps": type("Mps", (), {"is_available": staticmethod(lambda: False)})()})()
    monkeypatch.setitem(sys.modules, "torch", torch)

    assert _resolve_luxtts_device("cuda") == "cpu"


def test_luxtts_warm_and_release_share_the_provider_cache(tmp_path, fake_luxtts):
    from tools.tts_tool_lifecycle import release_tts_provider, warm_tts_provider

    reference = tmp_path / "consenting-speaker.wav"
    reference.write_bytes(b"RIFFfake")
    config = {"provider": "luxtts", "luxtts": {"ref_audio": str(reference), "device": "cpu"}}

    assert warm_tts_provider(config) == {
        "provider": "luxtts", "warmed": True, "action": "loaded", "elapsed_ms": pytest.approx(0, abs=1000),
    }
    assert release_tts_provider("luxtts") == {"released": 1}
