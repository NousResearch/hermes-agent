"""Tests: session.resume accepts ``inline_images=false``.

Attachment bytes are re-expanded to base64 on every history read, so a stored
transcript's payload is the sum of every attachment the conversation ever
carried, and ``session.resume`` (the first call after a reconnect) pays it
again each time. The reference-form branch already existed in
``_history_dict_text(content, image_urls=False)``; ``_coerce_message_text``
hard-coded ``image_urls=True``, so it was unreachable from the wire.

Contract:
- ``inline_images=false`` renders image parts as ``[image]``, not the data URI;
- the default (absent) keeps the data URI inlined — the local contract the
  desktop's ``extractEmbeddedImages`` depends on, so existing callers are
  byte-identical;
- the param reaches the history builder through ``_Resume``, and a non-true
  value reads as false.
"""

from __future__ import annotations

import tui_gateway.server as srv  # importing the facade installs the method_ctx bindings
from tui_gateway.methods_session import _Resume

# Production reaches the history builder through the server-bound copy (server.py rebinds these
# onto its own globals), so call it that way here rather than through the defining module.
_history_to_messages = srv._history_to_messages

# A tiny but recognisable data URI: the assertions are about whether the image
# BYTES travel, not about any particular image.
_DATA_URI = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="


def _history():
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": "look at this"},
            {"type": "image_url", "image_url": {"url": _DATA_URI}},
        ],
    }]


def test_default_keeps_the_image_data_uri_inlined():
    text = _history_to_messages(_history())[0]["text"]

    assert _DATA_URI in text
    assert "look at this" in text


def test_image_urls_false_renders_the_reference_form():
    text = _history_to_messages(_history(), image_urls=False)[0]["text"]

    assert _DATA_URI not in text
    assert "[image]" in text
    # The text part is untouched: only the image reference form changes.
    assert "look at this" in text


def test_resume_inlines_images_by_default():
    assert _Resume(1, {"session_id": "x"}, "x").inline_images is True


def test_resume_honours_inline_images_false():
    assert _Resume(1, {"session_id": "x", "inline_images": False}, "x").inline_images is False
