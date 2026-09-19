"""Token estimation for computer_use reports (RFC #112734).

Reports carry observation/delta *bytes*; providers bill *tokens*. Exact per-model
tokenization is not reproducible in-process, so every token count here is a
documented, deterministic estimate derived only from byte counts and pixel
dimensions — identical input always yields the identical estimate.

Text: 1 token per 4 UTF-8 bytes of English text, the long-standing BPE rule of
thumb (OpenAI documents ~4 characters per token). Empty input estimates 0.

Image: 85 base + 170 tokens per 512px tile after scaling to fit within 2048px and
then to a 768px short side — the published OpenAI high-detail vision schedule,
used as the reference grid because any per-tile vision pricer lands on the same
tiling. Other providers differ; the METHOD string stamped on each report says
which estimate the numbers are.
"""

from __future__ import annotations

import math

# Stamped on every report carrying these estimates so a reader can tell which
# method produced the numbers.
METHOD = "token_estimates.v1(text:1tok/4B; image:85+170/tile, 2048/768 prescale)"

_BYTES_PER_TOKEN = 4
_IMAGE_BASE_TOKENS = 85
_IMAGE_TILE_TOKENS = 170
_IMAGE_TILE_PX = 512
_IMAGE_LONG_SIDE_PX = 2048
_IMAGE_SHORT_SIDE_PX = 768


def estimate_text_tokens(nbytes: int) -> int:
    """Ceiling of bytes/4; 0 for empty input."""
    n = max(0, int(nbytes))
    return (n + _BYTES_PER_TOKEN - 1) // _BYTES_PER_TOKEN if n else 0


def estimate_image_tokens(width: int, height: int) -> int:
    """Tile-grid estimate for one screenshot; 0 when either dimension is missing."""
    w = float(max(0, int(width)))
    h = float(max(0, int(height)))
    if not w or not h:
        return 0
    down = min(1.0, _IMAGE_LONG_SIDE_PX / max(w, h))
    w, h = w * down, h * down
    up = _IMAGE_SHORT_SIDE_PX / min(w, h)
    w, h = w * up, h * up
    tiles = math.ceil(w / _IMAGE_TILE_PX) * math.ceil(h / _IMAGE_TILE_PX)
    return _IMAGE_BASE_TOKENS + _IMAGE_TILE_TOKENS * tiles
