"""``pdf.attach`` page rasterisation (issue #146639).

An iOS camera scan writes its MediaBox in *pixels as points* (2497x3467 pt =
34.7x48.1 in), so the old ``pdftoppm -png -r 150`` upscaled it ~2.08x into a ~40 MP
bitmap: minutes of poppler CPU and tens of MB of PNG for a 2-page, 4 MB scan.  The
handler now renders a long-edge-capped JPEG first and falls back to plain raster, then
to the PDF's embedded page images, all under the same 120 s budget the handler always
had.  These tests pin the rendering parameters, not the clock: output size and the
fallback order are deterministic, wall-clock bounds are not.
"""

from __future__ import annotations

import base64
import io
import subprocess
import sys
import threading
import types
from pathlib import Path

import pytest

from tui_gateway import server

pytest.importorskip("PIL")

_SCAN_W, _SCAN_H = 2497, 3467


def _build_scan_pdf(jpeg_bytes: bytes, width: int, height: int) -> bytes:
    """Single-page PDF whose MediaBox is ``width x height`` points, holding one JPEG.

    Poppler's broken-xref repair stands in for a valid object table.
    """
    contents = b"q %d 0 0 %d 0 0 cm /Im0 Do Q" % (width, height)
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d] "
            b"/Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>" % (width, height)
        ),
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(contents), contents),
        (
            b"<< /Type /XObject /Subtype /Image /Width 64 /Height 64 "
            b"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode "
            b"/Length %d >>\nstream\n%s\nendstream" % (len(jpeg_bytes), jpeg_bytes)
        ),
    ]
    out = bytearray(b"%PDF-1.4\n")
    for i, body in enumerate(objs, start=1):
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    out += b"xref\n0 1\ntrailer\n<< /Size 1 /Root 1 0 R >>\nstartxref\n0\n%%EOF\n"
    return bytes(out)


def _tiny_jpeg() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (240, 240, 240)).save(buf, "JPEG", quality=60)
    return buf.getvalue()


def _session() -> dict:
    return {
        "agent": types.SimpleNamespace(),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
    }


def _failing_run(reject):
    """subprocess.run stand-in: argv matching ``reject`` fails, everything else is real."""
    real_run = subprocess.run

    def fake_run(argv, *a, **kw):
        if reject(argv):
            return subprocess.CompletedProcess(argv, 1, "", "renderer unavailable here")
        return real_run(argv, *a, **kw)

    return fake_run


def _poppler_or_skip() -> None:
    import shutil

    if shutil.which("pdftoppm") is None or shutil.which("pdfimages") is None:
        pytest.skip("poppler-utils not installed")


@pytest.fixture()
def scan_pdf(tmp_path: Path) -> Path:
    """1 page, MediaBox 2497x3467 pt (i.e. a scan), ~0.1 MB."""
    _poppler_or_skip()
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(_build_scan_pdf(_tiny_jpeg(), _SCAN_W, _SCAN_H))
    return pdf


def _rpc(monkeypatch, tmp_path: Path, sid: str, params: dict) -> dict:
    fake_cli = types.ModuleType("cli")
    fake_cli._IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
    fake_cli._resolve_attachment_path = lambda raw: raw
    monkeypatch.setitem(sys.modules, "cli", fake_cli)
    monkeypatch.setattr(server, "_hermes_home", tmp_path / "home")
    server._sessions[sid] = _session()
    return server.handle_request(
        {"id": "1", "method": "pdf.attach", "params": {"session_id": sid, **params}}
    )


def _dims(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as img:
        return img.size


def test_scan_pages_are_capped_jpegs(monkeypatch, tmp_path: Path, scan_pdf: Path):
    """The pathological MediaBox comes out as a small JPEG, not a 40 MP PNG."""
    resp = _rpc(monkeypatch, tmp_path, "pdfcap", {"path": str(scan_pdf)})

    res = resp.get("result") or pytest.fail(f"error reply: {resp}")
    assert res["attached"] is True
    assert res["pages_attached"] == 1
    page = Path(res["pages"][0]["path"])
    assert page.suffix == ".jpg"
    assert max(_dims(page)) <= server._PDF_RENDER_CAP_PX


def test_falls_back_to_plain_raster_without_scale_to(monkeypatch, tmp_path: Path, scan_pdf: Path):
    """Old poppler (no ``-scale-to``): the plain 150 dpi attempt still delivers pages."""
    monkeypatch.setattr(subprocess, "run", _failing_run(lambda argv: "-scale-to" in argv))
    resp = _rpc(monkeypatch, tmp_path, "pdfplain", {"path": str(scan_pdf)})

    res = resp.get("result") or pytest.fail(f"error reply: {resp}")
    assert res["attached"] is True
    # Uncapped 150 dpi on a 2497x3467 pt page is a 2.08x upscale.
    assert max(_dims(Path(res["pages"][0]["path"]))) > server._PDF_RENDER_CAP_PX


def test_falls_back_to_embedded_images_when_rasteriser_fails(
    monkeypatch, tmp_path: Path, scan_pdf: Path
):
    """When pdftoppm cannot render the geometry, pdfimages still yields page pixels."""
    monkeypatch.setattr(
        subprocess, "run", _failing_run(lambda argv: Path(argv[0]).name == "pdftoppm")
    )
    resp = _rpc(monkeypatch, tmp_path, "pdfembed", {"path": str(scan_pdf)})

    res = resp.get("result") or pytest.fail(f"error reply: {resp}")
    assert res["pages_attached"] == 1
    assert _dims(Path(res["pages"][0]["path"]))[0] == 64  # the embedded image, verbatim


def test_base64_upload_uses_the_same_chain(monkeypatch, tmp_path: Path, scan_pdf: Path):
    """Remote clients send bytes; the chain must not depend on a host path."""
    resp = _rpc(
        monkeypatch,
        tmp_path,
        "pdfb64",
        {"content_base64": base64.b64encode(scan_pdf.read_bytes()).decode("ascii")},
    )
    assert resp["result"]["pages_attached"] == 1
    assert Path(resp["result"]["pages"][0]["path"]).suffix == ".jpg"
