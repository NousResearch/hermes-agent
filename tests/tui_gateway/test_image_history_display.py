"""Native image history projection regression for #120083."""

import base64

from tui_gateway import server


def test_verified_native_image_history_reconciles_without_inline_payload(tmp_path):
    image = tmp_path / "sample shot.png"
    image.write_bytes(b"image bytes")
    url = "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode()
    path = str(image)
    parts = [{"type": "image_url", "image_url": {"url": url}}]
    legacy = [{"type": "text", "text": f"Caption\n\n[Image attached at: {path}]"}, *parts,
              {"type": "text", "text": "<memory-context>private recall</memory-context>"}]
    current = [{"type": "text", "text": f"Caption\n@image:`{path}`"}, *parts]
    history = [{"role": "user", "content": legacy}, {"role": "assistant", "content": "Done"},
               {"role": "user", "content": current}]

    displayed = server._history_to_messages(history)
    assert displayed == [{"role": "user", "text": f"Caption\n\n@image:`{path}`"},
                         {"role": "assistant", "text": "Done"},
                         {"role": "user", "text": f"Caption\n@image:`{path}`"}]
    assert history[0]["content"] is legacy
    assert history[2]["content"] is current


def test_unavailable_or_mismatched_image_keeps_inline_payload(tmp_path):
    image = tmp_path / "shot.png"
    image.write_bytes(b"new bytes")
    url = "data:image/png;base64," + base64.b64encode(b"old bytes").decode()
    parts = [{"type": "text", "text": f"Caption\n@image:{image}"},
             {"type": "image_url", "image_url": {"url": url}}]
    row = {"role": "user", "content": parts}
    assert url in server._history_to_messages([row])[0]["text"]
    image.unlink()
    assert url in server._history_to_messages([row])[0]["text"]
