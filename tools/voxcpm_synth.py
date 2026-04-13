#!/usr/bin/env python3
"""Standalone VoxCPM synthesis helper.

Called by podcast_tool.py via subprocess to keep the TTS model (~8GB)
in a separate process that exits after synthesis — no lingering memory.

Supports three modes:
  1. Voice Design:    Text description → new voice (no reference audio)
  2. Controllable Cloning: Reference audio → cloned voice with style control
  3. Ultimate Cloning: Reference audio + transcript → highest fidelity clone

Usage:
    # Voice design (description only)
    python tools/voxcpm_synth.py \\
        --text "Hello world" \\
        --voice-description "A warm, authoritative male voice" \\
        --out output.wav

    # Voice cloning (reference audio)
    python tools/voxcpm_synth.py \\
        --text "Hello world" \\
        --reference-wav speaker.wav \\
        --out output.wav

    # Ultimate cloning (reference + transcript)
    python tools/voxcpm_synth.py \\
        --text "Hello world" \\
        --reference-wav speaker.wav \\
        --reference-text "Transcript of the reference audio." \\
        --out output.wav

Requires: pip install voxcpm (includes torch, torchaudio)
"""

import argparse
import struct
import sys
from pathlib import Path


def _write_wav(path: str, samples, sample_rate: int = 48000) -> None:
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


def _detect_device(requested: str) -> str:
    """Detect the best available device."""
    if requested != "auto":
        return requested

    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="VoxCPM synthesis helper")
    parser.add_argument("--text", required=True,
                        help="Text to synthesize")
    parser.add_argument("--out", required=True,
                        help="Output WAV path")
    parser.add_argument("--voice-description", default="",
                        help="Voice description for voice design mode "
                             "(e.g. 'A warm male voice, mid-30s')")
    parser.add_argument("--reference-wav", default="",
                        help="Reference audio path for voice cloning")
    parser.add_argument("--reference-text", default="",
                        help="Transcript of reference audio for ultimate cloning")
    parser.add_argument("--cfg-value", type=float, default=2.0,
                        help="Guidance strength (higher = more precise, default 2.0)")
    parser.add_argument("--inference-steps", type=int, default=10,
                        help="Diffusion steps (higher = better quality, default 10)")
    parser.add_argument("--device", default="auto",
                        help="Device: auto/cpu/cuda/mps (default: auto)")
    args = parser.parse_args()

    # Validate output directory
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate reference files if provided
    ref_wav = args.reference_wav
    ref_text = args.reference_text
    if ref_wav and not Path(ref_wav).expanduser().exists():
        print(f"Error: reference audio not found: {ref_wav}", file=sys.stderr)
        sys.exit(1)

    # Detect device
    device = _detect_device(args.device)
    print(f"INFO: Using device: {device}", file=sys.stderr)

    # Import VoxCPM
    try:
        from voxcpm import VoxCPM
    except ImportError:
        print("Error: voxcpm not installed. Run: pip install voxcpm",
              file=sys.stderr)
        sys.exit(1)

    # Load model
    print("INFO: Loading VoxCPM2 model...", file=sys.stderr)
    model = VoxCPM.from_pretrained(
        "openbmb/VoxCPM2",
        load_denoiser=False,
    )
    # Move to device if needed (the library handles this via from_pretrained
    # but we respect the device arg for MPS/CUDA selection)

    # Build the synthesis text with voice description prefix
    synth_text = args.text
    if args.voice_description and not ref_wav:
        # Voice design mode: prepend description in parentheses
        synth_text = f"({args.voice_description}){args.text}"

    # Choose synthesis mode
    generate_kwargs = {
        "text": synth_text,
        "cfg_value": args.cfg_value,
        "inference_timesteps": args.inference_steps,
    }

    if ref_wav and ref_text:
        # Ultimate cloning: reference + transcript + separate reference
        print("INFO: Ultimate cloning mode (reference + transcript)",
              file=sys.stderr)
        generate_kwargs["prompt_wav_path"] = str(Path(ref_wav).expanduser())
        generate_kwargs["prompt_text"] = ref_text
        generate_kwargs["reference_wav_path"] = str(Path(ref_wav).expanduser())
    elif ref_wav:
        # Controllable cloning: reference audio only
        print("INFO: Controllable cloning mode (reference audio)",
              file=sys.stderr)
        generate_kwargs["reference_wav_path"] = str(Path(ref_wav).expanduser())
    else:
        # Voice design: description in text prefix
        print("INFO: Voice design mode (text description)", file=sys.stderr)

    # Generate
    print(f"INFO: Synthesizing {len(args.text)} chars...", file=sys.stderr)
    wav = model.generate(**generate_kwargs)

    # Get sample rate from model
    sample_rate = getattr(model.tts_model, "sample_rate", 48000)

    # Write output
    try:
        import soundfile as sf
        sf.write(str(out_path), wav, sample_rate)
    except ImportError:
        _write_wav(str(out_path), wav, sample_rate)

    print(f"OK: {out_path} ({sample_rate}Hz)", file=sys.stderr)


if __name__ == "__main__":
    main()
