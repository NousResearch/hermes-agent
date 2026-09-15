"""Deterministic cleanup turning assistant Markdown into a spoken script.

Shared by explicit TTS calls, gateway auto-TTS, voice-mode streaming and the web
dashboard. Non-ASCII characters are written as escapes on purpose so the file
stays free of invisible/look-alike glyphs.
"""

from __future__ import annotations

import html
import re

# Sentinel appended to former heading lines so smooth_whitespace_for_tts folds the
# heading into the sentence after it ("Weather, it will be sunny") instead of a bare
# "Weather." label.
_HEAD = "\x00"

_MD_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:[^()]|\([^)]*\))*\)")
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\((?:[^()]|\([^)]*\))*\)")
_MD_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", flags=re.DOTALL)
_MD_UNDERSCORE_BOLD_RE = re.compile(r"__(.+?)__", flags=re.DOTALL)
_MD_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", flags=re.DOTALL)
_MD_UNDERSCORE_ITALIC_RE = re.compile(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", flags=re.DOTALL)
_MD_STRIKE_RE = re.compile(r"~~(.+?)~~", flags=re.DOTALL)
_MD_HEADING_LINE_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$", flags=re.MULTILINE)
_MD_BLOCKQUOTE_RE = re.compile(r"^\s*>\s?", flags=re.MULTILINE)
_MD_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", flags=re.MULTILINE)
_MD_HR_RE = re.compile(r"^\s*[-*_]{3,}\s*$", flags=re.MULTILINE)
_MD_TABLE_PIPE_RE = re.compile(r"\s*\|\s*")
_URL_RE = re.compile(r"https?://\S+")

_DEGREE_UNITS = (("C", "Celsius"), ("F", "Fahrenheit"))
# Unit suffix (regex, after a digit) -> spoken word; km/h variants before the bare "m".
_UNIT_WORDS = (
    (r"km\s*/\s*h", "kilometres per hour"), (r"km/h", "kilometres per hour"),
    (r"mm", "millimetres"), (r"cm", "centimetres"), (r"m", "metres"))
# Currency prefix (regex) -> spoken word; order matters (NZ$/A$/US$ before bare $).
_CURRENCY_WORDS = (
    (r"NZ\$", "New Zealand dollars", re.IGNORECASE), (r"A\$", "Australian dollars", re.IGNORECASE),
    (r"US\$", "US dollars", re.IGNORECASE), ("€", "euros", 0), ("£", "pounds", 0), (r"\$", "dollars", 0),
)

# Broad emoji / pictograph cleanup: most voice providers read emojis as awkward labels.
_EMOJI_RE = re.compile(
    "[\U0001F1E6-\U0001F1FF\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF☀-➿]+",
    flags=re.UNICODE)
_VARIATION_SELECTOR_RE = re.compile("[︎️]")

# --- Identifier normalization -------------------------------------------------
# Machine identifiers (hashes, API tokens, file names) are meant to be READ, not
# spoken. A listener needs a reference to the thing, not its characters. Rules
# are deliberately conservative: anything ambiguous is left alone.

# Hex strings of 12+ chars (git SHAs, hashes, UUID-minus-dashes).
_LONG_HEX_RE = re.compile(r"(?<![\w-])[0-9a-f]{12,}(?![\w-])", flags=re.IGNORECASE)
# Dashed hex identifiers (UUIDs): hex segments joined by single dashes, 32+ hex digits total.
_UUID_RE = re.compile(
    r"(?<![\w-])[0-9a-f]{8}(?:-[0-9a-f]{4,}){2,}(?![\w-])", flags=re.IGNORECASE)
# Opaque alnum tokens of 16+ chars (API keys, voice IDs, base62 IDs). Match
# broadly, then require upper AND lower AND digit in Python so normal words,
# acronyms, and dates never match.
_OPAQUE_TOKEN_RE = re.compile(r"(?<![\w-])[A-Za-z0-9]{16,}(?![\w-])")

_FILE_EXT_WORDS = {
    "png": "PNG", "jpg": "JPEG", "jpeg": "JPEG", "gif": "GIF", "svg": "S-V-G",
    "webp": "web-P", "pdf": "PDF", "doc": "Word", "docx": "Word",
    "xls": "spreadsheet", "xlsx": "spreadsheet", "csv": "C-S-V",
    "mp3": "M-P-3", "wav": "wave", "mp4": "M-P-4", "mov": "movie",
    "py": "Python", "js": "JavaScript", "ts": "TypeScript", "json": "Jason",
    "yaml": "YAML", "yml": "YAML", "toml": "toml", "md": "markdown",
    "zip": "zip", "tar": "tar", "gz": "gzipped", "html": "HTML",
}
# A file name: a non-empty stem, a dot, a known extension, word-boundary end.
# Stems ending in another known word (e.g. "e.g", sentence "end.") stay safe:
# the extension must be in the map AND the stem must look name-like
# (letter/digit start, no spaces, no sentence-ending punctuation).
_FILE_NAME_RE = re.compile(
    r"(?<![\w./-])(\w[\w.-]*?)\.(" + "|".join(_FILE_EXT_WORDS) + r")\b(?!\.\w)",
    flags=re.IGNORECASE)


def _spell_tail(token: str) -> str:
    return " ".join(token[-4:].lower())


def _sub_long_hex(match: re.Match) -> str:
    return f"hash ending in {_spell_tail(match.group(0))}"


def _sub_opaque_token(match: re.Match) -> str:
    token = match.group(0)
    # Opaque ID heuristic: needs the mix of cases AND a digit that real words
    # and acronyms never have ("internationalization", "NASA2026" stay safe).
    if not (any(c.isupper() for c in token) and any(c.islower() for c in token)
            and any(c.isdigit() for c in token)):
        return token
    return f"string ending in {_spell_tail(token)}"


def _sub_file_name(match: re.Match) -> str:
    stem, ext = match.group(1), match.group(2).lower()
    word = _FILE_EXT_WORDS[ext]
    # A meaningful stem stays ("report-2026 PDF file"); a purely numeric stem
    # carries no information, so drop it ("PNG file").
    stem_spoken = stem if re.search(r"[A-Za-z]", stem) else ""
    return f"{stem_spoken} {word} file".strip()


def normalize_identifiers_for_tts(text: str) -> str:
    """Rewrite machine identifiers into speakable references.

    Hashes/IDs become "ending in <last 4>", file extensions become words
    ("report.pdf" -> "report PDF file"), and underscores inside identifiers
    become pauses instead of a spoken "underscore". Conservative by design:
    short tokens, pure words, versions, and anything ambiguous pass through.
    """
    if not text:
        return ""
    text = _UUID_RE.sub(_sub_long_hex, text)
    text = _LONG_HEX_RE.sub(_sub_long_hex, text)
    text = _OPAQUE_TOKEN_RE.sub(_sub_opaque_token, text)
    text = _FILE_NAME_RE.sub(_sub_file_name, text)
    # Underscore-separated identifiers: "TTSS_2026" -> "TTSS 2026". Only inside
    # tokens with a letter or digit on BOTH sides, so paths (already stripped)
    # and standalone underscores never turn into stray words.
    return re.sub(r"(?<=[A-Za-z0-9])_(?=[A-Za-z0-9])", " ", text)


def strip_markdown_for_tts(text: str) -> str:
    """Strip Markdown/Telegram formatting while preserving readable words."""
    if not text:
        return ""
    text = html.unescape(str(text))
    text = _MD_CODE_BLOCK_RE.sub(" ", text)
    text = _MD_IMAGE_RE.sub(lambda m: f" {m.group(1)} " if m.group(1) else " ", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _URL_RE.sub("", text)
    text = _MD_INLINE_CODE_RE.sub(r"\1", text)
    text = _MD_BOLD_RE.sub(r"\1", text)
    text = _MD_UNDERSCORE_BOLD_RE.sub(r"\1", text)
    text = _MD_ITALIC_RE.sub(r"\1", text)
    text = _MD_UNDERSCORE_ITALIC_RE.sub(r"\1", text)
    text = _MD_STRIKE_RE.sub(r"\1", text)
    # Mark headings (do not just delete the marker): see _HEAD.
    text = _MD_HEADING_LINE_RE.sub(lambda m: m.group(1).rstrip() + _HEAD, text)
    text = _MD_BLOCKQUOTE_RE.sub("", text)
    text = _MD_LIST_ITEM_RE.sub("", text)
    text = _MD_HR_RE.sub("", text)
    # Leftover table pipes become pauses instead of a spoken "vertical bar".
    return _MD_TABLE_PIPE_RE.sub("; ", text)


def _normalize_temperature_ranges(text: str) -> str:
    """``11-17°C`` -> ``11 to 17 degrees Celsius`` (en/em dash or hyphen; unicode minus normalized)."""
    number = r"([-+\u2212]?\d+(?:\.\d+)?)"
    for unit, word in _DEGREE_UNITS:
        text = re.sub(
            r"(?<!\w)" + number + r"\s*[\u2013\u2014-]\s*" + number + r"\s*°\s*" + unit + r"\b",
            lambda m, w=word: (
                f"{m.group(1).replace(chr(0x2212), '-')} to {m.group(2).replace(chr(0x2212), '-')} degrees {w}"
            ),
            text, flags=re.IGNORECASE)
    return text


def normalize_symbols_for_tts(text: str) -> str:
    """Expand common symbols/shorthand into words a TTS engine reads well."""
    if not text:
        return ""
    text = re.sub("[   ]", " ", str(text))  # non-breaking / thin spaces
    text = text.replace("\u2212", "-").replace("…", "...")  # minus sign, ellipsis
    text = _normalize_temperature_ranges(text)
    # Temperatures with a number first, then bare units ("measured in degrees C"),
    # then any remaining degree symbol (angles, stray cases).
    for unit, word in _DEGREE_UNITS:
        text = re.sub(
            r"(?<!\w)([-+]?\d+(?:\.\d+)?)\s*°\s*" + unit + r"\b", r"\1 degrees " + word, text, flags=re.IGNORECASE,
        )
    for unit, word in _DEGREE_UNITS:
        text = re.sub(r"°\s*" + unit + r"\b", "degrees " + word, text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\w)([-+]?\d+(?:\.\d+)?)\s*°", r"\1 degrees", text).replace("°", " degrees")
    # Common weather/travel units.
    for pattern, word in _UNIT_WORDS:
        text = re.sub(r"(?<=\d)\s*" + pattern + r"\b", " " + word, text, flags=re.IGNORECASE)
    # Numeric rates only ("5/month" -> "5 per month").  Requiring digit-then-letter
    # keeps "and/or", "N/A", "TCP/IP" and dates like "2026/06" intact.
    text = re.sub(r"(?<=\d)\s*/\s*(?=[A-Za-z])", " per ", text)
    # Money and percentages. The integer part must END in a digit so a trailing
    # comma ("A$50, ...") is not swallowed into the spoken amount. Prefixed
    # currencies run first so "$" doesn't eat "NZ$".
    for symbol, word, flags in _CURRENCY_WORDS:
        text = re.sub(symbol + r"\s*([\d,]*\d(?:\.\d+)?)", r"\1 " + word, text, flags=flags)
    text = re.sub(r"(?<=\d)\s*%", " percent", text)
    # Operators and separators that commonly leak from formatted answers.
    text = re.sub("[•◦▪▫]", " ", text.replace("&", " and "))  # bullet glyphs
    for symbol, word in (("→", " to "), ("⇒", " to "), ("≈", " about "), ("~", " about ")):
        text = text.replace(symbol, word)
    return _EMOJI_RE.sub("", _VARIATION_SELECTOR_RE.sub("", text))


def smooth_whitespace_for_tts(text: str) -> str:
    """Collapse visual formatting into calm spoken paragraphs. A _HEAD-marked heading folds into
    the next content line as a lead-in ("Weather, It will be sunny."); a heading with no content
    after it becomes its own sentence."""
    if not text:
        return ""
    raw_lines = text.splitlines()
    add_sentence_pauses = sum(1 for raw_line in raw_lines if raw_line.replace(_HEAD, "").strip()) > 1
    lines: list[str] = []
    pending_heading: str | None = None

    def flush_pending() -> None:
        nonlocal pending_heading
        if pending_heading is not None:
            lines.append(pending_heading.rstrip(".:;,") + ".")
            pending_heading = None
    for raw_line in raw_lines:
        is_heading = raw_line.rstrip().endswith(_HEAD)
        line = raw_line.replace(_HEAD, "").strip()
        if not line:
            # Hold a pending heading across blank lines so it still folds into the next content line.
            if pending_heading is None and lines and lines[-1] != "":
                lines.append("")
            continue
        if is_heading:
            flush_pending()
            pending_heading = line.rstrip(".:;,")
            continue
        if pending_heading is not None:
            line = f"{pending_heading.rstrip('.:;,')}, {line}"
            pending_heading = None
        if add_sentence_pauses and line[-1] not in ".!?;:":
            line += "."
        lines.append(line)
    flush_pending()
    text = "\n".join(lines)
    for pattern, repl in ((r"\n{3,}", "\n\n"), (r"[ \t]{2,}", " "), (r"\s+([,.;:!?])", r"\1"),
                          (r"([,.;:!?])([A-Za-z])", r"\1 \2"), (r"\.{4,}", "...")):
        text = re.sub(pattern, repl, text)
    return text.strip()


# ``/reasoning show`` emits ``<think>...</think>`` in the final message: users want to
# SEE reasoning, not hear it. An unterminated block (streaming cut-off) is also silenced.
# Reasoning blocks: models with ``/reasoning show`` enabled emit ``<think>...</think>`` blocks in the final
# assistant message. See #34213.
_THINK_BLOCK_RE = re.compile(r"<think[\s>].*?</think>", flags=re.DOTALL | re.IGNORECASE)
_THINK_BLOCK_OPEN_RE = re.compile(r"<think[\s>].*\Z", flags=re.DOTALL | re.IGNORECASE)

# run_agent.py's turn-end file-mutation verifier footer (a ``⚠️ File-mutation verifier:``
# header line plus indented ``•`` bullets) is a UI affordance, not speech.
_VERIFIER_FOOTER_RE = re.compile(r"^\s*⚠️?\s*File-mutation verifier:.*(?:\n[ \t]+•.*)*", flags=re.MULTILINE)


def strip_nonspoken_blocks(text: str) -> str:
    """Remove ``<think>`` reasoning blocks and the file-mutation verifier footer."""
    if not text:
        return ""
    for pattern in (_THINK_BLOCK_RE, _THINK_BLOCK_OPEN_RE, _VERIFIER_FOOTER_RE):
        text = pattern.sub(" ", text)
    return text


def flatten_newlines_for_payload(text: str) -> str:
    """Collapse newlines into sentence breaks for single-line TTS payloads: some OpenAI-compatible
    backends (e.g. Kokoro) truncate at the first newline; smoothing already ends each line with
    punctuation, so this is safe.

    See #9004.
    """
    if not text:
        return ""
    for pattern, repl in ((r"\n{2,}", ". "), (r"(?<=[.!?;:,])\n", " "), (r"\n", ". "), (r"\.\s*\.", "."),
                          (r"[ \t]{2,}", " ")):
        text = re.sub(pattern, repl, text)
    return text.strip()


def prepare_spoken_text(text: str, max_chars: int | None = 4000) -> str:
    """Return a TTS-friendly script from assistant text (deterministic cleanup, not a rewrite).
    Pipeline: non-spoken blocks > Markdown > symbols/units > line formatting into sentence
    pauses > single line (for newline-sensitive providers), then ``max_chars``."""
    spoken = text
    for step in (strip_nonspoken_blocks, strip_markdown_for_tts, normalize_symbols_for_tts,
                 normalize_identifiers_for_tts, smooth_whitespace_for_tts,
                 flatten_newlines_for_payload):
        spoken = step(spoken)
    if max_chars is not None and max_chars > 0 and len(spoken) > max_chars:
        spoken = spoken[:max_chars].rstrip()
    return spoken


def _strip_markdown_for_tts(text: str) -> str:
    """``prepare_spoken_text`` without a length cap (``tts_tool`` compatibility name)."""
    return prepare_spoken_text(text, max_chars=None)
