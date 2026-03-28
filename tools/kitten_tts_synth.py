#!/usr/bin/env python3
"""Small helper that synthesizes speech with kittentts."""

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthesize speech with KittenTTS")
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--voice", default="Rosie")
    parser.add_argument("--model-name", default="KittenML/kitten-tts-nano-0.8")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--clean-text", action="store_true")
    args = parser.parse_args()

    from kittentts import KittenTTS

    model = KittenTTS(model_name=args.model_name, cache_dir=args.cache_dir)
    model.generate_to_file(
        args.text,
        args.out,
        voice=args.voice,
        speed=args.speed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
