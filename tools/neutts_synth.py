#!/usr/bin/env python3
"""Standalone NeuTTS synthesis helper.

Called by tts_tool.py via subprocess to keep the TTS model (~500MB)
in a separate process that exits after synthesis — no lingering memory.

Usage:
    python -m tools.neutts_synth --text "Hello" --out output.wav \
        --ref-audio samples/jo.wav --ref-text samples/jo.txt

Requires: python -m pip install -U neutts[all]
System:   apt install espeak-ng  (or brew install espeak-ng)
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
    samples = samples.flatten()

    # Clamp and convert to int16
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767).astype(np.int16)

    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
    block_align = num_channels * (bits_per_sample // 8)
    data_size = len(pcm) * (bits_per_sample // 8)

    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, num_channels, sample_rate,
                            byte_rate, block_align, bits_per_sample))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(pcm.tobytes())


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

    # Validate inputs
    ref_audio = Path(args.ref_audio).expanduser()
    ref_text_path = Path(args.ref_text).expanduser()
    if not ref_audio.exists():
        print(f"Error: reference audio not found: {ref_audio}", file=sys.stderr)
        sys.exit(1)
    if not ref_text_path.exists():
        print(f"Error: reference text not found: {ref_text_path}", file=sys.stderr)
        sys.exit(1)

    ref_text = ref_text_path.read_text(encoding="utf-8").strip()

    try:
        from neutts import NeuTTS
    except ImportError:
        raise ImportError(
            "neutts is required for local TTS synthesis. "
            "Install with: pip install hermes-agent[tts-local]"
        ) from None

    tts = NeuTTS(
        backbone_repo=args.model,
        backbone_device=args.device,
        codec_repo="neuphonic/neucodec",
        codec_device=args.device,
    )
    ref_codes = tts.encode_reference(str(ref_audio))
    wav = tts.infer(args.text, ref_codes, ref_text)

    # Write output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import soundfile as sf
    except ImportError:
        raise ImportError(
            "soundfile is required for audio output. "
            "Install with: pip install hermes-agent[tts-local]"
        ) from None
    sf.write(str(out_path), wav, 24000)

    print(f"OK: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
