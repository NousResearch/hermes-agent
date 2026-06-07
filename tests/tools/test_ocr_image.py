"""Tests for the verbatim OCR tool (tools/vision_tools.py::ocr_image_tool).

Covers:
- Successful OCR of a generated PNG with known text (mocked OpenRouter call) —
  asserts the known verbatim string is a substring of the returned text.
- Non-English (Latvian, diacritics) fixture is NOT translated — original tokens
  survive end-to-end.
- The forced OpenRouter contract (provider/model/base_url/temperature/prompt).
- Model/base_url resolution precedence (env > config > default).
- PDF rasterization: a tiny generated 1-page PDF is rasterized to data URLs
  (real pdf2image/poppler) and OCR'd page-by-page.
"""

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.vision_tools import (
    ocr_image_tool,
    _resolve_ocr_model,
    _resolve_ocr_base_url,
    _resolve_ocr_max_tokens,
    _rasterize_pdf,
    _is_pdf,
    _OCR_DEFAULT_MODEL,
    _OCR_DEFAULT_MAX_TOKENS,
    _OCR_MAX_TOKENS_CEILING,
    _OCR_PROMPT,
)


def _make_text_png(path: Path, text: str) -> Path:
    """Render `text` onto a real PNG using Pillow (default bitmap font)."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (640, 120), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 40), text, fill="black")
    img.save(path, format="PNG")
    return path


def _mock_llm_response(content: str, finish_reason: str = "stop") -> MagicMock:
    """Build a fake OpenAI-style chat completion carrying `content`.

    `finish_reason` defaults to "stop"; pass "length" to simulate a response
    truncated by the output-token budget.
    """
    resp = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    # Defang reasoning fallback so extract_content_or_reasoning takes content.
    choice.message.reasoning = None
    choice.finish_reason = finish_reason
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# Model / base_url resolution
# ---------------------------------------------------------------------------


class TestOcrResolution:
    def test_default_model(self):
        with patch("hermes_cli.config.load_config", return_value={}):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("HERMES_OCR_MODEL", None)
                assert _resolve_ocr_model() == _OCR_DEFAULT_MODEL

    def test_env_overrides_config(self):
        with patch("hermes_cli.config.load_config", return_value={
            "ocr": {"model": "from/config"}
        }):
            with patch.dict(os.environ, {"HERMES_OCR_MODEL": "from/env"}):
                assert _resolve_ocr_model() == "from/env"

    def test_config_model_used_when_no_env(self):
        with patch("hermes_cli.config.load_config", return_value={
            "ocr": {"model": "from/config"}
        }):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("HERMES_OCR_MODEL", None)
                assert _resolve_ocr_model() == "from/config"

    def test_base_url_default_is_openrouter(self):
        from hermes_constants import OPENROUTER_BASE_URL
        with patch("hermes_cli.config.load_config", return_value={}):
            assert _resolve_ocr_base_url() == OPENROUTER_BASE_URL

    def test_base_url_config_override(self):
        with patch("hermes_cli.config.load_config", return_value={
            "ocr": {"base_url": "https://example.test/v1"}
        }):
            assert _resolve_ocr_base_url() == "https://example.test/v1"

    def test_empty_base_url_falls_through_to_openrouter(self):
        # Finding 2: a misconfigured empty base_url must NOT slip through as "".
        from hermes_constants import OPENROUTER_BASE_URL
        with patch("hermes_cli.config.load_config", return_value={
            "ocr": {"base_url": ""}
        }):
            assert _resolve_ocr_base_url() == OPENROUTER_BASE_URL

    def test_whitespace_base_url_falls_through_to_openrouter(self):
        from hermes_constants import OPENROUTER_BASE_URL
        with patch("hermes_cli.config.load_config", return_value={
            "ocr": {"base_url": "   "}
        }):
            assert _resolve_ocr_base_url() == OPENROUTER_BASE_URL

    def test_empty_model_falls_through_to_default(self):
        with patch("hermes_cli.config.load_config", return_value={
            "ocr": {"model": ""}
        }):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("HERMES_OCR_MODEL", None)
                assert _resolve_ocr_model() == _OCR_DEFAULT_MODEL

    def test_whitespace_env_model_falls_through(self):
        with patch("hermes_cli.config.load_config", return_value={}):
            with patch.dict(os.environ, {"HERMES_OCR_MODEL": "   "}):
                assert _resolve_ocr_model() == _OCR_DEFAULT_MODEL


class TestOcrMaxTokens:
    def test_default(self):
        with patch("hermes_cli.config.load_config", return_value={}):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("HERMES_OCR_MAX_TOKENS", None)
                assert _resolve_ocr_max_tokens() == _OCR_DEFAULT_MAX_TOKENS

    def test_config_override(self):
        with patch("hermes_cli.config.load_config", return_value={
            "ocr": {"max_tokens": 8000}
        }):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("HERMES_OCR_MAX_TOKENS", None)
                assert _resolve_ocr_max_tokens() == 8000

    def test_env_overrides_config(self):
        with patch("hermes_cli.config.load_config", return_value={
            "ocr": {"max_tokens": 8000}
        }):
            with patch.dict(os.environ, {"HERMES_OCR_MAX_TOKENS": "5000"}):
                assert _resolve_ocr_max_tokens() == 5000

    def test_clamped_to_ceiling(self):
        with patch("hermes_cli.config.load_config", return_value={
            "ocr": {"max_tokens": 10_000_000}
        }):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("HERMES_OCR_MAX_TOKENS", None)
                assert _resolve_ocr_max_tokens() == _OCR_MAX_TOKENS_CEILING

    def test_empty_and_invalid_fall_through_to_default(self):
        for bad in ("", "   ", "abc", "0", "-5"):
            with patch("hermes_cli.config.load_config", return_value={
                "ocr": {"max_tokens": bad}
            }):
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("HERMES_OCR_MAX_TOKENS", None)
                    assert _resolve_ocr_max_tokens() == _OCR_DEFAULT_MAX_TOKENS, bad


# ---------------------------------------------------------------------------
# Image OCR — success, verbatim substring, no translation
# ---------------------------------------------------------------------------


class TestOcrImageSuccess:
    @pytest.mark.asyncio
    async def test_known_text_is_substring(self, tmp_path):
        known = "INVOICE 2026 Total: $42.00"
        img = _make_text_png(tmp_path / "doc.png", "rendered")

        with patch(
            "tools.vision_tools.async_call_llm",
            new_callable=AsyncMock,
            return_value=_mock_llm_response(known),
        ):
            result = json.loads(await ocr_image_tool(str(img)))

        assert result["success"] is True
        assert result["pages"] == 1
        assert known in result["text"]

    @pytest.mark.asyncio
    async def test_non_english_not_translated(self, tmp_path):
        # Latvian with diacritics — model must NOT translate; tokens survive.
        latvian = "Šodien ir ļoti silts. Ēdiens gārša ātrāk."
        img = _make_text_png(tmp_path / "lv.png", "rendered")

        with patch(
            "tools.vision_tools.async_call_llm",
            new_callable=AsyncMock,
            return_value=_mock_llm_response(latvian),
        ):
            result = json.loads(await ocr_image_tool(str(img)))

        assert result["success"] is True
        # Original Latvian tokens present (no English translation substituted).
        for token in ("Šodien", "ļoti", "Ēdiens", "gārša"):
            assert token in result["text"], f"missing verbatim token {token!r}"
        # Sanity: a translation would contain these English words.
        lowered = result["text"].lower()
        assert "today" not in lowered
        assert "food" not in lowered

    @pytest.mark.asyncio
    async def test_forced_openrouter_contract(self, tmp_path):
        img = _make_text_png(tmp_path / "c.png", "x")

        with patch("hermes_cli.config.load_config", return_value={}):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("HERMES_OCR_MAX_TOKENS", None)
                os.environ.pop("HERMES_OCR_MODEL", None)
                with patch(
                    "tools.vision_tools.async_call_llm",
                    new_callable=AsyncMock,
                    return_value=_mock_llm_response("text"),
                ) as mock_llm:
                    await ocr_image_tool(str(img))

        kw = mock_llm.await_args.kwargs
        assert kw["task"] == "vision"
        assert kw["provider"] == "openrouter"
        assert kw["model"] == _OCR_DEFAULT_MODEL
        assert kw["temperature"] == 0
        assert kw["max_tokens"] == _OCR_DEFAULT_MAX_TOKENS
        assert kw["timeout"] == 120
        # Strict verbatim prompt is sent.
        text_part = kw["messages"][0]["content"][0]["text"]
        assert text_part == _OCR_PROMPT
        assert "Do NOT translate" in text_part

    @pytest.mark.asyncio
    async def test_explicit_model_override_argument(self, tmp_path):
        img = _make_text_png(tmp_path / "m.png", "x")
        with patch(
            "tools.vision_tools.async_call_llm",
            new_callable=AsyncMock,
            return_value=_mock_llm_response("text"),
        ) as mock_llm:
            await ocr_image_tool(str(img), model="custom/vl")
        assert mock_llm.await_args.kwargs["model"] == "custom/vl"


# ---------------------------------------------------------------------------
# Truncation signaling (finding 3)
# ---------------------------------------------------------------------------


class TestOcrTruncation:
    @pytest.mark.asyncio
    async def test_not_truncated_by_default(self, tmp_path):
        img = _make_text_png(tmp_path / "ok.png", "x")
        with patch(
            "tools.vision_tools.async_call_llm",
            new_callable=AsyncMock,
            return_value=_mock_llm_response("short text", finish_reason="stop"),
        ):
            result = json.loads(await ocr_image_tool(str(img)))
        assert result["success"] is True
        assert result["truncated"] is False

    @pytest.mark.asyncio
    async def test_truncated_flag_set_on_length(self, tmp_path):
        img = _make_text_png(tmp_path / "long.png", "x")
        with patch(
            "tools.vision_tools.async_call_llm",
            new_callable=AsyncMock,
            return_value=_mock_llm_response("cut off", finish_reason="length"),
        ):
            result = json.loads(await ocr_image_tool(str(img)))
        assert result["success"] is True
        assert result["truncated"] is True

    @pytest.mark.asyncio
    async def test_pdf_truncated_flag_set_on_length(self, tmp_path):
        pdf = _make_one_page_pdf(tmp_path / "trunc.pdf", "rendered")
        with patch(
            "tools.vision_tools.async_call_llm",
            new_callable=AsyncMock,
            return_value=_mock_llm_response("partial page", finish_reason="length"),
        ):
            result = json.loads(await ocr_image_tool(str(pdf)))
        assert result["success"] is True
        assert result["truncated"] is True

    @pytest.mark.asyncio
    async def test_configured_max_tokens_passed_to_llm(self, tmp_path):
        img = _make_text_png(tmp_path / "mt.png", "x")
        with patch("hermes_cli.config.load_config", return_value={
            "ocr": {"max_tokens": 9000}
        }):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("HERMES_OCR_MAX_TOKENS", None)
                with patch(
                    "tools.vision_tools.async_call_llm",
                    new_callable=AsyncMock,
                    return_value=_mock_llm_response("text"),
                ) as mock_llm:
                    await ocr_image_tool(str(img))
        assert mock_llm.await_args.kwargs["max_tokens"] == 9000


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestOcrImageErrors:
    @pytest.mark.asyncio
    async def test_missing_file_returns_error(self, tmp_path):
        result = json.loads(await ocr_image_tool(str(tmp_path / "nope.png")))
        assert result["success"] is False
        assert result["text"] == ""

    @pytest.mark.asyncio
    async def test_non_image_file_rejected(self, tmp_path):
        secret = tmp_path / "secret.txt"
        secret.write_text("not an image")
        with patch(
            "tools.vision_tools.async_call_llm", new_callable=AsyncMock
        ) as mock_llm:
            result = json.loads(await ocr_image_tool(str(secret)))
        assert result["success"] is False
        mock_llm.assert_not_called()


# ---------------------------------------------------------------------------
# PDF rasterization (real pdf2image + poppler)
# ---------------------------------------------------------------------------


def _make_one_page_pdf(path: Path, text: str) -> Path:
    """Build a tiny 1-page PDF by rendering text to PNG then wrapping it."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (612, 200), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 80), text, fill="black")
    img.save(path, format="PDF")  # Pillow can emit a single-page PDF
    return path


class TestPdfRasterization:
    def test_is_pdf_detection(self, tmp_path):
        pdf = _make_one_page_pdf(tmp_path / "doc.pdf", "hi")
        assert _is_pdf(pdf) is True
        png = _make_text_png(tmp_path / "doc.png", "hi")
        assert _is_pdf(png) is False

    def test_rasterize_one_page(self, tmp_path):
        pdf = _make_one_page_pdf(tmp_path / "one.pdf", "PAGE ONE")
        data_urls = _rasterize_pdf(pdf)
        assert len(data_urls) == 1
        assert data_urls[0].startswith("data:image/png;base64,")

    @pytest.mark.asyncio
    async def test_pdf_ocr_concatenates_pages(self, tmp_path):
        pdf = _make_one_page_pdf(tmp_path / "ocr.pdf", "rendered")
        with patch(
            "tools.vision_tools.async_call_llm",
            new_callable=AsyncMock,
            return_value=_mock_llm_response("PAGE TEXT"),
        ):
            result = json.loads(await ocr_image_tool(str(pdf)))
        assert result["success"] is True
        assert result["pages"] == 1
        assert "PAGE TEXT" in result["text"]

    @pytest.mark.asyncio
    async def test_file_uri_pdf_routes_to_pdf_path(self, tmp_path):
        # Finding 6: a file:///tmp/doc.pdf-style input strips the scheme and
        # routes to the PDF rasterization path (pages>=1), not the image path.
        pdf = _make_one_page_pdf(tmp_path / "doc.pdf", "rendered")
        file_uri = "file://" + str(pdf)
        with patch(
            "tools.vision_tools.async_call_llm",
            new_callable=AsyncMock,
            return_value=_mock_llm_response("PAGE TEXT"),
        ) as mock_llm:
            result = json.loads(await ocr_image_tool(file_uri))
        assert result["success"] is True
        assert result["pages"] == 1
        assert "PAGE TEXT" in result["text"]
        # The PDF path must actually invoke the OCR LLM (not be rejected).
        mock_llm.assert_awaited()
