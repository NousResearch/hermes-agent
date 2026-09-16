#!/usr/bin/env python3
"""Standalone NeuTTS synthesis helper.

Called by tts_tool via subprocess so the ~500MB TTS model lives in a process that exits
after synthesis. Usage:
    python -m tools.neutts_synth --text "Hello" --out out.wav --ref-audio jo.wav --ref-text jo.txt
Requires ``pip install -U neutts[all]`` and espeak-ng (apt/brew).
"""

import argparse
import struct
import sys
from pathlib import Path


def _write_wav(path: str, samples, sample_rate: int = 24000) -> None:
    """Write a WAV file from float32 samples (no soundfile dependency)."""
    import numpy as np
    if not isinstance(samples, np.ndarray):
        samples = np.array(samples, dtype=np.float32)
    pcm = (np.clip(samples.flatten(), -1.0, 1.0) * 32767).astype(np.int16)
    data_size = len(pcm) * 2  # 16-bit mono
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36 + data_size) + b"WAVEfmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
        f.write(b"data" + struct.pack("<I", data_size) + pcm.tobytes())


def main():
    parser = argparse.ArgumentParser(description="NeuTTS synthesis helper")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--out", required=True, help="Output WAV path")
    parser.add_argument("--ref-audio", required=True, help="Reference voice audio path")
    parser.add_argument("--ref-text", required=True, help="Reference voice transcript path")
    parser.add_argument("--model", default="neuphonic/neutts-air-q4-gguf",
                        help="HuggingFace backbone model repo")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda/mps)")
    args = parser.parse_args()

    ref_audio = Path(args.ref_audio).expanduser()
    ref_text_path = Path(args.ref_text).expanduser()
    for label, p in (("audio", ref_audio), ("text", ref_text_path)):
        if not p.exists():
            print(f"Error: reference {label} not found: {p}", file=sys.stderr)
            sys.exit(1)
    ref_text = ref_text_path.read_text(encoding="utf-8").strip()

    try:
        from neutts import NeuTTS
    except ImportError:
        print("Error: neutts not installed. Run: python -m pip install -U neutts[all]", file=sys.stderr)
        sys.exit(1)

    # llama_cpp (backbone) offloads to GPU only for the literal string "gpu";
    # torch (codec) only accepts "cuda". A single --device value can't satisfy
    # both — "cuda" silently no-ops on the backbone, leaving it on CPU.
    tts = NeuTTS(
        backbone_repo=args.model,
        backbone_device="gpu" if args.device == "cuda" else args.device,
        codec_repo="neuphonic/neucodec",
        codec_device=args.device)
    wav = tts.infer(args.text, tts.encode_reference(str(ref_audio)), ref_text)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf
        sf.write(str(out_path), wav, 24000)
    except ImportError:
        _write_wav(str(out_path), wav, 24000)
    print(f"OK: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
